import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

#计算单个类别的分割评估指标pred预测结果，gt是真实分割结果
def calculate_metric_percase(pred, gt):
    #将预测结果和真是标签中的非0值设置为1，确保他们是二值化的
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    #检查预测结果是否为空，sum是计算预测中1的像素总数
    if pred.sum() > 0:
        #计算预测结果和真是标签之间的dice系数和%95hausdorff距离，衡量两个点集中%95的点之间的最大距离
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

#对单个体积图像进行分割评估，接收四个参数：输入图像，image，对应的标签label，神经网络net， 类别数calsses
def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    #移除输入图像的第一个维度（假设是批量大小），将他们从gpu转移到cpu，并从pytorch张量转换为numpy数组
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    #创建一个与标签形状相同的numpy数组prediction，用于存储预测结果
    prediction = np.zeros_like(label)
    #遍历输入图像的每个切片（在第一个维度上）
    for ind in range(image.shape[0]):
        #从输入图像提取第ind个切片
        slice = image[ind, :, :]
        #获取当前切片大小
        x, y = slice.shape[0], slice.shape[1]
        #使用zoom函数调整切片的大小，使其匹配patchsize指定的尺寸
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        #将调整后的图像转换为pytorch张量，并增加维度，移动到cuda中
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        #将网络设置为评估模式，关闭dropout和batchnorm等训练特有的层
        net.eval()
        #前向传播，关闭梯度计算，减少内存占用和计算量
        with torch.no_grad():
            #net(input)将输入张量传递给神经网络，得到输出，torch.softmax对输出进行softmax，计算每个像素属于每个类别的概率
            #torch.argmax(取概率最大的类别作为预测结果，squeeze(0)移除批量大小维度，形状变为[h,w]，dim=1表示在类别维度上进行操作
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            #将结果从gpu转移到cpu，并转换为numpy数组
            out = out.cpu().detach().numpy()
            #将预测结果调整回原始切片大小
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            #将调整后的预测结果存储到prediction数组中
            prediction[ind] = pred
    #计算评估指标，存储每个类别的评估指标
    metric_list = []
    #遍历每个类别，跳过背景类别0
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_one_model(image, label, net, classes, patch_size=[256, 256], output_index=0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            #首先加载这个模型
            out = net(input)
            
            #筛选出是哪一个输出,如果是双输出模型
            if isinstance(out, tuple):
                #提取其中一个模型的输出
                out = out[output_index]
            else:
                out = out
            
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list