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

BATCH_SIZE = 16
NUM_EPOCHS = 50
#文件读取
test_rootPath = "work/test/"
train_rootPath = "work/train_val/" #文件夹目录
train_label = pd.read_csv('work/train_val.csv')

files = Series.as_matrix(train_label["name"])
label = Series.as_matrix(train_label["lable"])
files = [i+".npz" for i in files]
fileTest= os.listdir(test_rootPath)
fileTest.sort(key=lambda x:int(x[9:-4]))

fileNum = len(files)
trainNum = int(fileNum*0.85)
fileTrain = files[:trainNum]
fileVali = files[trainNum:]
labTrain = label[:trainNum]
labVali = label[trainNum:]

#define prepoecessor
def preprocess(pile):
    result = pile['voxel'] * pile['seg']
    result = result[34:66,34:66,34:66]  #裁剪到32*32*32
    result = result/255                 #假的归一化
    result = result*2-1                 #假的归一化到-1~1
    result = torch.from_numpy(result)   #转换为tensor
    result = torch.tensor(result,dtype=torch.float32)
    return result

def testprocess(pile):
    result = pile['voxel']*pile['seg']
    result = result[18:82,18:82,18:82]
    result = result/255                 #假的归一化
    result = result*2-1                 #假的归一化到-1~1
    result = torch.from_numpy(result)   #转换为tensor
    result = torch.tensor(result,dtype=torch.float32)
    return result

#define train Dataloader
def default_loader(path):
    pile = np.load(path)
    #print(voxel.shape)
    tensor = preprocess(pile)          #裁剪，归一化
    return tensor

def test_loader(path):
    pile = np.load(path)
    tensor = testprocess(pile)
    return tensor

#自定义dataset
class trainSet(Dataset):
    def __init__(self,loader=default_loader):
        self.images = fileTrain         #训练集文件地址列表
        self.target = labTrain
        self.loader = loader
    def __getitem__(self,index):
        path = self.images[index]
        img = self.loader(train_rootPath+path)
        target = self.target[index]
        return img,target
    def __len__(self):
        return len(self.images)
#define Vali Dataloader
class valiSet(Dataset):
    def __init__(self,loader=default_loader):
        self.images = fileVali         #评估集文件地址列表
        self.target = labVali
        self.loader = loader
    def __getitem__(self,index):
        path = self.images[index]
        img = self.loader(train_rootPath+path)
        target = self.target[index]
        return img,target
    def __len__(self):
        return len(self.images)
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

train_data = trainSet()
train_loader = DataLoader(train_data,batch_size = BATCH_SIZE,
                          shuffle=True,drop_last=True)
vali_data = valiSet()
vali_loader = DataLoader(vali_data,batch_size=1,
                         shuffle=False, drop_last=False)
test_data = testSet()
test_loader = DataLoader(test_data,batch_size=1,
                         shuffle=False, drop_last=False)
########################数据载入结束#########################

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

model = DenseNet(drop_rate=0.3)
#define loss function and optimiter
criterion = nn.BCELoss()
optimizer = optim.Adadelta(model.parameters(),lr = 0.001)

train_losses = np.empty([0,])
train_accuracy = np.empty([0,])
vali_losses = np.empty([0,])
vali_accuracy = np.empty([0,])

for epoch in range(NUM_EPOCHS):
    
    train_loss = 0
    train_accu = 0
    train_correct = 0
    vali_loss = 0
    vali_accu = 0
    vali_correct = 0
    
    optimizer.zero_grad()
    
    
    for images, labels in tqdm(train_loader):
        labels = torch.tensor(labels,dtype=torch.float32)
        #images = images.view(len(images),1,32,32,32)
        images = images.cuda()
        labels = labels.cuda()
        model.cuda().train()
        images = randomVerify(images)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        #计算每一batch的loss和accuracy
        for i in range(len(output)):
            if(output[i]>0.5):
                output[i] = 1.
            else:
                output[i] = 0.
        train_correct += torch.eq(output,labels).sum().item()
        train_loss += float(loss.item())
        #train_accu = train_correct/BATCH_SIZE
    
    #计算每一epoch的loss和accuracy并显示
    train_accu = train_correct/len(train_loader)/BATCH_SIZE
    Loss = train_loss/len(train_loader)
    print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,Loss,train_accu))
    train_losses = np.append(train_losses,Loss)
    train_accuracy = np.append(train_accuracy,train_accu)
    
    for images,labels in tqdm(vali_loader):
        labels = torch.tensor(labels,dtype=torch.float32)
        #images = images.view(BATCH_SIZE,1,32,32,32)
        images = images.cuda()
        labels = labels.cuda()
        model.cuda().eval()
        output = model(images)
        
        loss = criterion(output, labels)
        #计算每一batch的loss和accuracy
        for i in range(len(output)):
            if(output[i]>0.5):
                output[i] = 1.
            else:
                output[i] = 0.
        vali_correct += torch.eq(output,labels).sum().item()
        vali_loss += float(loss.item())
        #train_accu = train_correct/BATCH_SIZE
    
    #计算每一epoch的loss和accuracy并显示
    vali_accu = vali_correct/len(vali_loader.dataset)
    Loss = vali_loss/len(vali_loader.dataset)
    print("Vali Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,Loss,vali_accu))
    vali_losses = np.append(vali_losses,Loss)
    vali_accuracy = np.append(vali_accuracy,vali_accu)
    
    if(vali_accu > 0.65):
        torch.save(model,'model1Epoch%d.pt'%(epoch+1))

del train_loader,vali_loader

#TTA无标签集预测方法
def test():
    model.cuda().eval()
    out = np.empty([0,])
    for images in tqdm(test_loader):
    #images = images.view(BATCH_SIZE,1,32,32,32)
        images = images.cuda()
        images1 = images[:,16:48,16:48,16:48]
        """
        images2 = images[:,0:32,0:32,0:32]
        images3 = images[:,0:32,32:64,32:64]
        images4 = images[:,0:32,0:32,32:64]
        images5 = images[:,0:32,32:64,0:32]
        images6 = images[:,32:64,32:64,32:64]
        images7 = images[:,32:64,0:32,0:32]
        images8 = images[:,32:64,0:32,32:64]
        images9 = images[:,32:64,32:64,0:32]
        output1 = model(images1)
        output2 = model(images2)
        output3 = model(images3)
        output4 = model(images4)
        output5 = model(images5)
        output6 = model(images6)
        output7 = model(images7)
        output8 = model(images8)
        output9 = model(images9)
        output = (output1+output2+output3+output4+output5+output6+output7+output8+output9)/9
        """
        output = model(images1)
        out = np.append(out,output.item())
    return out

pred = test()
pred = pred.reshape([-1,1])
np.savetxt('pred2.csv',pred)

"""
#卷积池化层
def FeatureLayer(in_channel,out_channel,kernel_size=3,stride=1):
    layer = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,bias=False),
            #因为BN接收的层数输入是conv层的输出，所以feature number为out_channel
            nn.BatchNorm3d(num_features=out_channel), 
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=3))
    return layer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.layer1 = FeatureLayer(1,25)                #一次卷积
        self.layer1 = FeatureLayer(1,50,3,1)
        self.layer2 = FeatureLayer(50,100,3,1)              #二次卷积
        self.layer3 = nn.Dropout3d(p=0.5)
        self.layer4 = nn.Linear(800,100)      #这里出错呃原因可能是Linear层失去了batch特性
        self.layer5 = nn.Linear(100,1)
        self.layer6 = nn.ReLU6()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print(x.shape)
        x = x.view(BATCH_SIZE,-1)         #转变维度时一定注意不能丧失BATCH_SIZE信息
        x = self.layer4(x)
        x = self.layer5(x)
        #x = nn.Dropout3D(x, p=0.5, training=self.training)
        x = self.layer6(x)
        x /= 6
        return x
    
model = Net()
#define loss function and optimiter
criterion = nn.BCELoss()
optimizer = optim.Adadelta(model.parameters())

# train and evaluate
train_accuracy = np.array([1,NUM_EPOCHS])
vali_accuracy = np.array([1,NUM_EPOCHS])
#count = 0
for epoch in range(NUM_EPOCHS):
    print(epoch)
    optimizer.zero_grad()
    for images, labels in train_loader:
        #count += 1
        images = images.view(BATCH_SIZE,1,32,32,32)
        output = model(images)
        #print(output)
        labels = torch.tensor(labels,dtype=torch.float32)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        #print(images,labels)

vali_result = np.empty([0,1])

for d, target in vali_loader:
    d = d.view(BATCH_SIZE,1,32,32,32)
    pred = model(d)
    vali = pred.detach().numpy()
    vali_result = np.vstack((vali_result,vali))
    #print(pred)
    target = torch.tensor(target,dtype=torch.float32)
    #print(d,target)
for d, target in train_loader:
    d = d.view(BATCH_SIZE,1,32,32,32)
    pred = model(d)
    #print(pred)
    target = torch.tensor(target,dtype=torch.float32)
"""
# download and load the data
#train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
#Vali_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)
#由于没有调用torchvision.dataset,数据不会被自动trandform,需要后续实现
# encapsulate them into dataloader form
#trainVoxel_loader = data.DataLoader(voxTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#trainSeg_loader = data.DataLoader(segTrain, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
#训练集
"""
voxTrain = np.empty([trainNum,100,100,100])
segTrain = np.empty([trainNum,100,100,100])
for idx in range(len(fileTrain)): #遍历文件夹
    file = fileTrain[idx]
    print(str(file)[9:])
    train_data = np.load(path+'/'+str(file)+'.npz')
    voxTrain[idx,:,:,:] = train_data['voxel']
    segTrain[idx,:,:,:] = train_data['seg']
"""
#先只将voxel作为训练数据传入网络，不传seg
"""
np.save(temp_dir+'/trainData.npy',voxTrain)
np.save(temp_dir+'/trainLabel.npy',labTrain)
print("training set seperated")
#测试集
voxVali = np.empty([fileNum-trainNum,100,100,100])
segVali = np.empty([fileNum-trainNum,100,100,100])
for idx in range(len(fileVali)):
    file = fileVali[idx]
    print(str(file)[9:])
    Vali_data = np.load(path+'/'+str(file)+'.npz')
    voxVali[idx,:,:,:] = Vali_data['voxel']
    segVali[idx,:,:,:] = Vali_data['seg']
"""
#先只将voxel作为测试数据传入网络，不传seg
"""
np.save(temp_dir+'/ValiData.npy',voxVali)
np.save(temp_dir+'/ValiLabel.npy',labVali)
print("Valiing set seperated")
"""
