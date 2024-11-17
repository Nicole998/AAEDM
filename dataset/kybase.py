from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from sklearn.model_selection import train_test_split

# transform = transforms.Compose([
#     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
# ])
'''NPY数据格式'''


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, mode, transform = None):
        self.mode = mode
        self.data_6 = np.load(data)  # 加载npy数据
        # print('work')#one for traning, one for testing
        self.altitude = self.data_6[:,:,:,0]
        self.speed = self.data_6[:, :, :, 1]
        self.predicted = self.data_6[:, :, :, 3]
        self.data_3 = np.stack((self.altitude, self.speed, self.predicted), axis=3)
        self.all_labels = np.load(labels).astype(int)  #加载npy标签
       
        X_train,X_test,y_train,y_test = train_test_split(self.data_3,self.all_labels,test_size=0.2,random_state=10,shuffle=False)
        if self.mode == 'train':
            self.data = X_train
            self.labels = y_train
        elif self.mode == 'eval':
            self.data = X_test
            self.labels = y_test
      
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