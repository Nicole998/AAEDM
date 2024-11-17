# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch


# class Generator(nn.Module):
#     def __init__(self, out_dim=512, normalize_output=False):
#         super(Generator, self).__init__()
#         self.out_dim = out_dim
#         self.normalize_output = normalize_output
#
#         self.fc1 = nn.Linear(self.out_dim * 3, self.out_dim)
#         nn.init.normal_(self.fc1.weight, std=0.02)
#         # self.l0.weight.data.copy_(???)
#         # w = chainer.initializers.Normal(0.02)
#
#         self.fc2 = nn.Linear(self.out_dim, self.out_dim)
#         nn.init.normal_(self.fc2.weight, std=0.02)
#         # copy weight to l1
#         # self.l1.weight.data.copy_(???)
#
#     def forward(self, x):
#         fc1_out = self.fc1(x)
#         fc2_out = self.fc2(F.relu(fc1_out))
#         if self.normalize_output:
#             fc2_out_norm = torch.norm(fc2_out, p=2, dim=1, keepdim=True)
#             fc2_out = fc2_out / fc2_out_norm.expand_as(fc2_out)
#         return fc2_out

class Generator(nn.Module):
    def __init__(self, out_dim=512, normalize_output=False):
        super(Generator, self).__init__()
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        self.fc1 = nn.Linear(self.out_dim, 1024)
        nn.init.normal_(self.fc1.weight, std=0.02)
        # self.l0.weight.data.copy_(???)
        # w = chainer.initializers.Normal(0.02)

        self.fc2 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc2.weight, std=0.02)

        self.fc3 = nn.Linear(2048, 1024)
        nn.init.normal_(self.fc3.weight, std=0.02)

        # copy weight to l1
        # self.l1.weight.data.copy_(???)
        self.fc4 = nn.Linear(1024, self.out_dim)
        nn.init.normal_(self.fc4.weight, std=0.02)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(F.relu(fc1_out))
        fc3_out = self.fc3(F.relu(fc2_out))
        fc4_out = self.fc4(F.relu(fc3_out))
        if self.normalize_output:
            fc4_out_norm = torch.norm(fc4_out, dim=1, keepdim=True)
            fc4_out = fc4_out / fc4_out_norm.expand_as(fc4_out)
        return fc4_out


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, normalize_output=True):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        self.fc1 = nn.Linear(self.in_dim, 1024)
        nn.init.eye_(self.fc1.weight)

        self.fc2 = nn.Linear(1024, 2048)
        nn.init.eye_(self.fc2.weight)

        self.fc3 = nn.Linear(2048, 1024)
        nn.init.eye_(self.fc3.weight)
        
        self.fc4 = nn.Linear(1024, self.out_dim)
        nn.init.eye_(self.fc4.weight)
        

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(F.relu(fc1_out))
        fc3_out = self.fc3(F.relu(fc2_out))
        fc4_out = self.fc4(F.relu(fc3_out))
        if self.normalize_output:
            fc4_out_norm = torch.norm(fc4_out, p=2, dim=1, keepdim=True)
            fc4_out = fc4_out / fc4_out_norm.expand_as(fc4_out)
        return fc4_out
      
class Discriminator2(nn.Module):
    def __init__(self, in_dim, out_dim, normalize_output=True):
        super(Discriminator2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        self.model = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.out_dim),
        )
        #初始化权重
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.model(x)
        if self.normalize_output:
            out_norm = torch.norm(out, p=2, dim=1, keepdim=True)
            out = out / out_norm.expand_as(out)
        return out
