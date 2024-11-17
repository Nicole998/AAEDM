import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split,TensorDataset
# from model.Resnet34 import resnet34

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.labels = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}

        # if self.transform:
        #     sample['data'] = self.transform(sample['data'])

        return sample


# data_transform = {
#     "train": transforms.Compose([
#         transforms.ToTensor(),
#     ]),
#     "val": transforms.Compose([
#         transforms.ToTensor(),
#     ])
# }
# 定义划分比例
train_ratio = 0.7
test_ratio = 0.3

data_x = np.load('201116-X_all(label_noise_remove_0.3,size3605).npy')
data_y = np.load('201116-Y_all(label_noise_remove_0.3,size3605).npy')
data_x = data_x.transpose((0, 3, 1, 2))
print(data_x.shape)  # (3605, 6, 173, 173)

# 计算训练集和测试集大小
train_size = int(train_ratio * len(data_x))
test_size = len(data_x) - train_size

# dataset = MyDataset(data_x, data_y, transform=data_transform["train"])
dataset = MyDataset(data_x, data_y, transform=None)

# 划分训练集和测试集
# total_size = len(dataset)
# train_size = int(0.7 * total_size)
# test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # 2523 1082
print(train_dataset[0]['data'].shape)  # torch.Size([6, 173, 173])
# 保存为文件
np.save('train_dataset2523.npy',train_dataset)
np.save('test_dataset1082.npy',test_dataset)

# batch_size = 16  # 适当选择一个合适的批次大小
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # for data in test_loader:
# #     imgs,targets = data['data'],data['label']
# #     print(imgs.shape) # torch.Size([16, 6, 173, 173])
# #     print(targets.shape) # torch.Size([16])
# #     break
# # model = resnet34(pretrained=False)  # 如果你有预训练的权重，可以设为True
# model = resnet34()  # 如果你有预训练的权重，可以设为True
# # model.to(torch.double)
#
# num_classes = 5  # 你的数据集有5个类别
# model.fc = nn.Linear(model.fc.in_features, num_classes)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 5  # 适当选择训练的轮数
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# for epoch in range(num_epochs):
#     model.train()
#     for batch in train_loader:
#         images, labels = batch['data'].to(device,dtype=torch.float), batch['label'].to(device,dtype=torch.long)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     # 在这里添加验证集上的性能评估代码，可选
#
# # 训练结束后，可以保存模型
# torch.save(model.state_dict(), 'resnet34_model.pth')
# # 测试模型
# model.eval()
# correct = 0
# total = 0
#
# with torch.no_grad():
#     for batch in test_loader:
#         images, labels = batch['data'].to(device,dtype=torch.float), batch['label'].to(device,dtype=torch.float)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = correct / total
# print(f'Test Accuracy: {accuracy * 100:.2f}%')


# for data in test_loader:
#     imgs,targets = data
#     print(imgs)
#     print(targets)
#     break
