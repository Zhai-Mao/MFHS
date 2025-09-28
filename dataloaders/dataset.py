import os
# import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

#继承dataset这个类,这个类通常用于加载和处理数据集，支持多种数据增强操作
#base_dir:数据集的根目录路径， split：数据集的分割方式，通常为“train”。“val”或“test”， num“：加载的数据数量，如果为none，则加载全部数据
#transform：数据转换操作，包含一系列数据预处理或增强操作
#ops_weak:弱增强操作，用于自监督或半监督中，对原始数据进行轻微的抖动，例如随机翻转、随即裁剪、随机亮度调整
#ops_strong:强增强，用于自监督或半监督操作，对原始数据进行更剧烈的扰动
class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        val_root_path=None,
    ):
        self._base_dir = base_dir
        #存储与当前实例相关的数据样本、文件路径或其他信息，样本列表
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.val_root_path = val_root_path
        
        #确保ops_weak和ops_strong的存在性是一致的，错误信息表示如果要使用CTAugment学习到的策略，必须同时提供弱增强和强增强操作
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"
        
        #根据数据集的分割类型split，架子啊训练数据的文件路径列表，如果是train，会从一个指定的文件中读取训练数据的文件路劲，并将路径存储到self。sample_list中
        if self.split == 'train':
            with open(self._base_dir + "/train_slices.list", "r") as f1:
               #读取文件中的所有行
                self.sample_list = f1.readlines() 
            #移出每个字符串末尾的换行符
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        
        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))
        
    def __len__(self):
        return len(self.sample_list)
    
    #根据索引idx获取数据集中的一个样本，这个方法从存储数据的HDF5文件中读取图像和标签
    def __getitem__(self, idx):
        #从sample_list中获取索引为为idx的样本名称
        case = self.sample_list[idx]
        #使用h5py库打开HDF5文件
        if self.split == "train":
            #（）中的是完整路径
            h5f = h5py.File(self._base_dir + "/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self.val_root_path + "/{}.h5".format(case), "r")
        
        #从HDF5文件中读取图像和标签数据，[:]：表示读取整个数据集的内容
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        #对一个样本进行数据增强
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        
        sample["idx"] = idx   
        return sample

    
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image
    
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
    
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
    
#这里会对批次中整理一定要包含有标签数据和无标签数据
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices #主数据索引[0,1,2,3]
        self.secondary_indices = secondary_indices #无标签数据索引 [a,b,c]
        self.secondary_batch_size = secondary_batch_size #无标签批次大小 1
        self.primary_batch_size = batch_size - secondary_batch_size #有标签批次

        assert len(self.primary_indices) >= self.primary_batch_size > 0 
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0
    
    #初始化迭代器
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )
    
    #每个批次有多少个样本
    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

#打乱顺序，只遍历一轮
def iterate_once(iterable):
    return np.random.permutation(iterable)

#无限次训练，每次遍历前打乱顺序
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

#从索引中取
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
        
    #迭代器方法，有标签数据迭代一次，无标签数据无限循环遍历索引（每次遍历前随机排列）