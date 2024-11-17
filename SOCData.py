from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import PIL.Image
import torch
import random
import numpy as np

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

def inception_transform(is_train):
   
    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
       [
        RGBToBGR(),
        transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(inception_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities([0, 1], [0, 255]),
        transforms.Normalize(mean=inception_mean, std=inception_std)
       ])
    return inception_transform



class SOCDataset(torch.utils.data.Dataset):
    def __init__(self,data, labels, transform = None):
        self.data_6=data
        self.labels=labels
        self.altitude = self.data_6[:,:,:,0]
        self.speed = self.data_6[:, :, :, 1]
        self.predicted = self.data_6[:, :, :, 3]
        self.data_3 = np.stack((self.altitude, self.speed, self.predicted), axis=3)
        self.data=self.data_3
        self.transform = transform  # 转为tensor形式

    def __getitem__(self, index):
        img = self.data[index, :, :, :]  # 读取每一个npy的数据
        label = self.labels[index]
        
        img = PIL.Image.fromarray(np.uint8(img))  # 转成image的形式
        #img = img.convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)# 转为tensor形式
        
        return img, label  # 返回数据还有标签

    def __len__(self):
        return self.data.shape[0]  # 返回数据的总个数
      
    def nb_classes(self):
    #    assert set(self.labels) == set(self.classes)
        return 5