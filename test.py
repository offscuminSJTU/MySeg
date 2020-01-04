# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:25:19 2019

@author: dell
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
from pandas import Series
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
#import torch.backends.cudnn as cudnn
import random
import torch.cuda as cuda

test_rootPath = "work/test/"
model_Path = "model1Epoch40.pt"
save_Path = "Submission.csv"
fileTest= os.listdir(test_rootPath)
fileTest.sort(key=lambda x:int(x[9:-4]))

#define prepoecessor

def testprocess(pile):
    result = pile['voxel']*pile['seg']
    result = result[18:82,18:82,18:82]
    result = result/255                 #假的归一化
    result = result*2-1                 #假的归一化到-1~1
    result = torch.from_numpy(result)   #转换为tensor
    result = torch.tensor(result,dtype=torch.float32)
    return result

def test_loader(path):
    pile = np.load(path)
    tensor = testprocess(pile)
    return tensor

#define Test Dataloader
class testSet(Dataset):
    def __init__(self,loader=test_loader):
        self.images = fileTest         #测试集文件地址列表
        self.loader = loader
    def __getitem__(self,index):
        path = self.images[index]
        img = self.loader(test_rootPath+path)
        return img
    def __len__(self):
        return len(self.images)

test_data = testSet()
test_loader = DataLoader(test_data,batch_size=1,shuffle=False, drop_last=False)

class DenseLayer(nn.Sequential):
    def __init__(self,in_channel,growth_rate,bn_size,drop_rate): #bn_size为层内参数，表示层内运算中feature map扩大的比率
        super(DenseLayer,self).__init__()
        self.add_module('norm1',nn.BatchNorm3d(in_channel)),
        self.add_module('relu1',nn.ReLU(inplace=True)),
        self.add_module('conv1',nn.Conv3d(in_channel,bn_size*growth_rate,
                                          kernel_size=1,stride=1,bias=False)),
        self.add_module('norm2',nn.BatchNorm3d(bn_size*growth_rate)),
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv3d(bn_size*growth_rate,growth_rate,
                                          kernel_size=3,stride=1,padding=1,bias=False))
        self.drop_rate = drop_rate
    def forward(self,x):
        new_feature = super(DenseLayer,self).forward(x)
        if(self.drop_rate>0):
            new_feature = F.dropout3d(new_feature,p=self.drop_rate,training=self.training)
        return torch.cat([x,new_feature],1) #每一层denselayer的输出为该层的输入和输出的堆叠

class DenseBlock(nn.Sequential):
    def __init__(self,layer_num,in_channel,bn_size,growth_rate,drop_rate):
        super(DenseBlock,self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channel+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('denselayer%d'%(i+1),layer)

class Transition(nn.Sequential):
    def __init__(self,in_channel,zip_ratio=0.5):
        super(Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm3d(in_channel))
        self.add_module('relu',nn.ReLU(inplace=True))
        #需要在每一个Dense block之后接transition层用1*1conv将channel再拉回到一个相对较低的值（一般为输入的一半）
        self.add_module('conv',nn.Conv3d(in_channel,int(in_channel*zip_ratio),
                                         kernel_size=1,stride=1,bias=False))
        self.add_module('pool',nn.AvgPool3d(kernel_size=2,stride=2))

#卷积池化层
def FeatureLayer(in_channel,out_channel):
    layer = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=3,bias=False),
            #因为BN接收的层数输入是conv层的输出，所以feature number为out_channel
            nn.BatchNorm3d(num_features=out_channel), 
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=3,stride=2,padding=1))
    return layer

class DenseNet(nn.Module):
    """
    growth_rate:每个DenseLayer输出的feature map个数
    block_config:每个DenseBlock中DenseLayer的个数
    in_channel:输入feature map个数
    bn_size:DenseLayer中feature map的扩大率
    drop_rate:Dropout3d参数
    """
    def __init__(self,growth_rate=32,block_config=(4,4,4),init_channel_num=64,
                 bn_size=4,drop_rate=0):
        super(DenseNet,self).__init__()
        self.feature_layer = FeatureLayer(1,init_channel_num)
        #增加DenseBlock与Transition
        channel_num = init_channel_num
        for i,layer_num in enumerate(block_config):
            block = DenseBlock(layer_num=layer_num,in_channel=channel_num,
                               bn_size=bn_size,growth_rate=growth_rate,
                               drop_rate=drop_rate)
            self.feature_layer.add_module('denseblock%d'%(i+1),block)
            channel_num = channel_num + layer_num*growth_rate
            #对于非最后一级的denseblock，增加transition层
            if(i!=len(block_config)-1):
                trans = Transition(channel_num,0.5)
                self.feature_layer.add_module('transition%d'%(i+1),trans)
                channel_num = int(0.5*channel_num)
        #增加Classifier
        self.feature_layer.add_module('norm5',nn.BatchNorm3d(channel_num))
        self.feature_layer.add_module('relu5',nn.ReLU(inplace=True))
        self.feature_layer.add_module('avgpool5',
                                      nn.AvgPool3d(kernel_size=3,stride=2))
        self.classifier = nn.Linear(channel_num,1) #Linear层的输入个数尚不清楚
        self.sigmoid = torch.sigmoid
        
        # Official init from torch repo.
        # 8太知道这一段是干嘛的
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
    def forward(self,x):
        x = x.view(len(x),1,32,32,32)
        features = self.feature_layer(x)
        out = features.view(len(x),-1)
        out = self.classifier(out)
        out = self.sigmoid(out)
        out = out.view(len(out))
        return out

#对输入的数据进行翻转
def reflect(images):
    for i in range(len(images)):
        ref = random.randint(0,1)
        if(ref == 1):
            axis = random.randint(0,2) # 0:x轴翻转，1:y轴翻转，2:z轴翻转
            center = 16 #循环中不包括16
            if(axis == 0):
                for a in range(center): #0~15
                    images[i,a,:,:],images[i,31-a,:,:] = images[i,31-a,:,:],images[i,a,:,:]
            elif(axis == 1):
                for a in range(center):
                    images[i,:,a,:],images[i,:,31-a,:] = images[i,:,31-a,:],images[i,:,a,:]
            elif(axis == 2):
                for a in range(center):
                    images[i,:,:,a],images[i,:,:,31-a] = images[i,:,:,31-a],images[i,:,:,a]
        else:
            continue
    return images

def rotate(images):
    for i in range(len(images)):
        rot = random.randint(0,1)
        if(rot == 1):
            axis = random.randint(0,2) # 0:x轴为旋转轴,1:y轴为旋转轴,2:z轴为旋转轴
            if(axis == 0):
                for a in range(32):
                    images[i,a,:,:] = torch.transpose(images[i,a,:,:], 0, 1)
            if(axis == 1):
                for a in range(32):
                    images[i,:,a,:] = torch.transpose(images[i,:,a,:], 0, 1)
            if(axis == 2):
                for a in range(32):
                    images[i,:,:,a] = torch.transpose(images[i,:,:,a], 0, 1)
        else:
            continue
    return images

def randomVerify(images):
    images = reflect(images)
    images = rotate(images)
    return images

###########################模型定义结束#########################

model = DenseNet()
model.load_state_dict(torch.load(model_Path))

#TTA无标签集预测方法
def test():
    model.cuda().eval()
    out = np.empty([0,])
    for images in tqdm(test_loader):
    #images = images.view(BATCH_SIZE,1,32,32,32)
        images = images.cuda()
        images1 = images[:,16:48,16:48,16:48]
        output = model(images1)
        out = np.append(out,output.item())
    return out

pred = test()
pred = pred.reshape([-1,])
for i in range(len(fileTest)):
    fileTest[i] = fileTest[i][:-4]
df = pd.DataFrame({'Id': fileTest,'Predicted': pred})
df.to_csv(save_Path)