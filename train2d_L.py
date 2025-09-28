# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
import os
import csv
import argparse
import sys
import time
import random
import logging
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from utils.config import get_config
from utils import losses, metrics, ramps
from model.net_factory import net_factory
from model.Discriminator import Discriminator, FCDiscriminator
from val_2D import (test_single_volume, test_single_volume_one_model)
from dataloaders.dataset import (BaseDataSets, RandomGenerator,TwoStreamBatchSampler)

sys.path.append("/root/autodl-tmp") 


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='ACDC/data/slices', help='name of Experiment')
parser.add_argument('--val_root_path', type=str, default='ACDC/data',help='name of val path')
parser.add_argument('--exp', type=str, default="TransHyper", help="experiment_name")
parser.add_argument('--model', type=str, default='SAM2UNetL', help='model_name')
parser.add_argument('--max_iterations', type=int, default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.1, help=', segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224],help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--opts', help="modify config options by adding key_value pairs", default=None, nargs='+')
parser.add_argument('--zip',action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode',type=str, default='part', choices=['no', 'full', 'part'], help='no:no cache,''full:cache all data,''part:sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help='gradient accumulation steps')
parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpoint to save memmory')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput',action='store_true', help='Test throughput only')
parser.add_argument('--labeled_bs',type=int, default=8, help='labeled batch_size_per gpu')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=int, default=200, help='consistency_rampup')

#discriminator
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
parser.add_argument("--lambda-adv-target1", type=float, default=0.001,
                        help="lambda_adv for adversarial training.")
parser.add_argument("--lambda-adv-target2", type=float, default=0.001,
                        help="lambda_adv for adversarial training.")

parser.add_argument("--hiera_path", type=str, default="sam2_hiera_large.pt", 
                    help="path to the sam2 pretrained hiera")

args = parser.parse_args()
config = get_config(args)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(optimizer, i_batch):
    lr = lr_poly(args.lr, i_batch, args.max_iterations, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    lr_ds = args.lr
    
    model_D1 = FCDiscriminator(num_classes=1)
    model_D2 = FCDiscriminator(num_classes=1)
    model_D3 = FCDiscriminator(num_classes=1)
    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()
    
    model_D3.train()
    model_D3.cuda()
    
    def create_model(ema=False):     
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, checkpoint_path=args.hiera_path)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,   
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val", val_root_path=args.val_root_path)

    total_slices = len(db_train)

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
    
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model.train()
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()
    
    optimizer_D3 = optim.Adam(model_D3.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_D3.zero_grad()
    
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    results_path = os.path.join(snapshot_path, 'results_train_2d_one_M_shape_model_notensorboard_mutual_train_discriminator_softmax_FCdiscriminatorv2_3decoder_add_modeltrain_encoder2hyper_10thres1_iteration40000_1hyper_SAM2UNet__largerbase_lr0_1_3')
    os.makedirs(results_path, exist_ok=True)
    scalar_csv_path = os.path.join(results_path, 'scalars.csv')
    with open(scalar_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'LR', 'Consistency Weight', 'Total Loss'])
    
    images_path = os.path.join(results_path, 'images')
    os.makedirs(images_path, exist_ok=True)
    
    validation_csv_path = os.path.join(results_path, 'validation.csv')
    with open(validation_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Class', 'Dice', 'HD95'])
    
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    best_performance3 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    source_label = 0
    target_label = 1
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_D3.zero_grad()
            
            adjust_learning_rate_D(optimizer_D1, i_batch)
            adjust_learning_rate_D(optimizer_D2, i_batch)
            adjust_learning_rate_D(optimizer_D3, i_batch)
            
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False
            
            for param in model_D3.parameters():
                param.requires_grad = False
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = Variable(volume_batch).cuda(), label_batch.cuda()
            
            last_output, penultimate_output1, penultimate_output2 = model(volume_batch)
            last_soft = torch.softmax(last_output, dim=1)
            penultimate_soft1 = torch.softmax(penultimate_output1, dim=1)
            penultimate_soft2 = torch.softmax(penultimate_output2, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            
            labeled_loss1 = 0.5 * (
                ce_loss(last_output[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + 
                dice_loss(last_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            )
            labeled_loss2 = 0.5 * (
                ce_loss(penultimate_output1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + 
                dice_loss(penultimate_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            )
            labeled_loss3 = 0.5 * (
                ce_loss(penultimate_output2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + 
                dice_loss(penultimate_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            )
            
            with torch.no_grad():
                pseudo_labels1 = torch.argmax(last_soft[args.labeled_bs:], dim=1)
                pseudo_labels2 = torch.argmax(penultimate_soft1[args.labeled_bs:], dim=1)
                pseudo_labels3 = torch.argmax(penultimate_soft2[args.labeled_bs:], dim=1)
            
            unlabeled_loss1 = dice_loss(
                last_soft[args.labeled_bs:], 
                pseudo_labels2.unsqueeze(1)
            )
            unlabeled_loss11 = dice_loss(
                last_soft[args.labeled_bs:], 
                pseudo_labels3.unsqueeze(1)
            )
            unlabeled_loss2 = dice_loss(
                penultimate_soft1[args.labeled_bs:], 
                pseudo_labels1.unsqueeze(1)
            )
            unlabeled_loss21 = dice_loss(
                penultimate_soft1[args.labeled_bs:], 
                pseudo_labels3.unsqueeze(1)
            )
            unlabeled_loss3 = dice_loss(
                penultimate_soft2[args.labeled_bs:], 
                pseudo_labels1.unsqueeze(1)
            )
            unlabeled_loss31 = dice_loss(
                penultimate_soft2[args.labeled_bs:], 
                pseudo_labels2.unsqueeze(1)
            )
            
            adv_weight = 0.1  
            model1_loss = labeled_loss1 + consistency_weight * unlabeled_loss1 + consistency_weight * unlabeled_loss11
            model2_loss = labeled_loss2 + consistency_weight * unlabeled_loss2 + consistency_weight * unlabeled_loss21
            model3_loss = labeled_loss3 + consistency_weight * unlabeled_loss3 + consistency_weight * unlabeled_loss31
            loss = model1_loss + model2_loss + model3_loss
            loss.backward()
            
            volume_batch2, label_batch2 = sampled_batch['image'], sampled_batch['label']
            volume_batch2, label_batch2 = Variable(volume_batch2).cuda(), label_batch2.cuda()
            
            last_output2, penultimate_output21l, penultimate_output22l = model(volume_batch2)
            last_soft2 = torch.softmax(last_output2, dim=1)
            penultimate_soft21 = torch.softmax(penultimate_output21l, dim=1)
            penultimate_soft22 = torch.softmax(penultimate_output22l, dim=1)
            
            
            last_output_max_indices = torch.argmax(last_soft2[:args.labeled_bs], dim=1, keepdim=True).float()
            penultimate_output1_max_indices = torch.argmax(penultimate_soft21[:args.labeled_bs], dim=1, keepdim=True).float()
            penultimate_output2_max_indices = torch.argmax(penultimate_soft22[:args.labeled_bs], dim=1, keepdim=True).float()
            last_output_max_indices.requires_grad = True
            penultimate_output1_max_indices.requires_grad = True
            penultimate_output2_max_indices.requires_grad = True
            D_out1 = model_D1(last_output_max_indices)
            D_out2 = model_D2(penultimate_output1_max_indices)
            D_out3 = model_D3(penultimate_output2_max_indices)
            
            loss_adv_target1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
            loss_adv_target2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())
            loss_adv_target3 = bce_loss(D_out3, Variable(torch.FloatTensor(D_out3.data.size()).fill_(target_label)).cuda())
            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2 + args.lambda_adv_target2 * loss_adv_target3
            loss.backward()
            
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True
                
            for param in model_D3.parameters():
                param.requires_grad = True
            
            label_batch_ = label_batch[:args.labeled_bs].float().unsqueeze(1)
            D_out1x = model_D1(label_batch_)
            D_out2x = model_D2(label_batch_)
            D_out3x = model_D3(label_batch_)
            loss_D1 = bce_loss(D_out1x, Variable(torch.FloatTensor(D_out1x.data.size()).fill_(target_label)).cuda())
            loss_D2 = bce_loss(D_out2x, Variable(torch.FloatTensor(D_out2x.data.size()).fill_(target_label)).cuda())
            loss_D3 = bce_loss(D_out3x, Variable(torch.FloatTensor(D_out3x.data.size()).fill_(target_label)).cuda())
            
            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            last_output_max_indices1 = torch.argmax(last_soft[:args.labeled_bs].detach(), dim=1, keepdim=True).float()
            penultimate_output_max_indices11 = torch.argmax(penultimate_soft1[:args.labeled_bs].detach(), dim=1, keepdim=True).float()
            penultimate_output_max_indices12 = torch.argmax(penultimate_soft2[:args.labeled_bs].detach(), dim=1, keepdim=True).float()
            D_out1l = model_D1(last_output_max_indices1)
            D_out2l = model_D2(penultimate_output_max_indices11)
            D_out3l = model_D3(penultimate_output_max_indices12)
            loss_D1 = bce_loss(D_out1l, Variable(torch.FloatTensor(D_out1l.data.size()).fill_(source_label)).cuda())
            loss_D2 = bce_loss(D_out2l, Variable(torch.FloatTensor(D_out2l.data.size()).fill_(source_label)).cuda())
            loss_D3 = bce_loss(D_out3l, Variable(torch.FloatTensor(D_out3l.data.size()).fill_(source_label)).cuda())
            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_D3.step()
            
            iter_num = iter_num + 1
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            with open(scalar_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([iter_num, lr_, consistency_weight, loss.item()])
            
            logging.info('iteration %d : model1 loss : %f model2 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item(), model3_loss.item()))
            
            if iter_num % 50 == 0:
                try:
                    with torch.no_grad():
                        image = volume_batch[1, 0:1, :, :].cpu().numpy()
                        image_path = os.path.join(images_path, f'train_Image_{iter_num}.png')
                        image = (image * 255).astype(np.uint8)
                        image = image.squeeze()
                        Image.fromarray(image).save(image_path)

                        outputs = torch.argmax(torch.softmax(last_output, dim=1), dim=1, keepdim=True)
                        pred = outputs[1, 0:1, :, :].cpu().numpy()  # 通过 [1, 0] 降维到 (H, W)
                        pred = (pred * 50).astype(np.uint8)
                        
                        pred = pred.squeeze()
                        output_path = os.path.join(images_path, f'train_model1_Prediction_{iter_num}.png')
                        Image.fromarray(pred).save(output_path)

                        outputs = torch.argmax(torch.softmax(penultimate_output1, dim=1), dim=1, keepdim=True)
                        pred = outputs[1, 0:1, :, :].cpu().numpy()  # 通过 [1, 0] 降维到 (H, W)
                        pred = (pred * 50).astype(np.uint8)
                        
                        pred = pred.squeeze()
                        output_path = os.path.join(images_path, f'train_model2_Prediction_{iter_num}.png')
                        Image.fromarray(pred).save(output_path)
                        
                        outputs = torch.argmax(torch.softmax(penultimate_output2, dim=1), dim=1, keepdim=True)
                        pred = outputs[1, 0:1, :, :].cpu().numpy()  # 通过 [1, 0] 降维到 (H, W)
                        pred = (pred * 50).astype(np.uint8)
                        
                        pred = pred.squeeze()
                        output_path = os.path.join(images_path, f'train_model3_Prediction_{iter_num}.png')
                        Image.fromarray(pred).save(output_path)

                        labs = label_batch[1, ...].unsqueeze(0).cpu().numpy() * 50
                        labs = labs.squeeze()
                        label_path = os.path.join(images_path, f'train_GroundTruth_{iter_num}.png')
                        Image.fromarray(labs.astype(np.uint8)).save(label_path)
                except Exception as e:
                    logging.warning(f"保存第 {iter_num} 次迭代的图像失败: {str(e)}")
                    logging.info(f"volume_batch 形状: {volume_batch.shape}")
                    logging.info(f"last_output 形状: {last_output.shape}")
                    logging.info(f"label_batch 形状: {label_batch.shape}")
           
            if iter_num > 0 and iter_num % 200 ==0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_one_model(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size, output_index=0)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                
                for class_i in range(num_classes-1):
                    with open(validation_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([iter_num, class_i+1, metric_list[class_i, 0], metric_list[class_i, 1]])
                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                with open(validation_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([iter_num, 'mean', performance1, mean_hd951])
                
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_one_Mmodel_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model_one_Mmodel.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                #..........................................................................................................#
                metric_list_penultimate = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_one_model(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size, output_index=1)
                    metric_list_penultimate += np.array(metric_i)
                metric_list_penultimate = metric_list_penultimate / len(db_val)
                
                for class_i in range(num_classes-1):
                    with open(validation_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([iter_num, class_i+1, metric_list_penultimate[class_i, 0], metric_list_penultimate[class_i, 1]])
                performance2 = np.mean(metric_list_penultimate, axis=0)[0]

                mean_hd951 = np.mean(metric_list_penultimate, axis=0)[1]
                with open(validation_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([iter_num, 'mean', performance2, mean_hd951])
                
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    
                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd951))
                model.train()
                
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
            
            time1 = time.time()
            
        if iter_num >= max_iterations:
            iterator.close()
            break
            
    writer.close()
            
            
                
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    snapshot_path = "work_dir/{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path + '/log.txt', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)
    








