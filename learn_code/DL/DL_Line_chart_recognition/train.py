# -*- coding: unicode_escape -*-
"cxking"
# 加载torch模块
import torch
#数据处理模块
import torchvision
import torchvision.transforms as transforms
#模型优化模块
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 其他模块
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 #过滤无用的警告
import os
import warnings
warnings.filterwarnings('ignore')

# 数据路径
root='/home/cxking/datasets/line_data_new/'
# 数据处理class
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,root, datatxt, transform=None, target_transform=None): #初始化一些需要传入的参数
        print(os.path.join(root,datatxt))
        fh = open(root + datatxt, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []                      #创建一个名为img的空列表，一会儿用来装东西
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()   #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                                        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    #用于按照索引读取每个元素的具体内容
    def __getitem__(self, index):    
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(root+fn).convert('RGB') #按照path读入图片from PIL import Image # 按照路径读取图片
 
        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img,label  #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
 


train_transform = transforms.Compose([  #打包各操作函数
    transforms.RandomRotation(10),  #以指定的角度选装图片。（-10，10）旋转角度范围
    transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,概率为0.5。即：一半的概率翻转，一半的概率不翻转。
    transforms.Resize(224),
    transforms.CenterCrop(224), #将给定的PIL.Image进行中心切割，得到给定的size，size可以是tuple，(target_height, target_width)。size也可以是一个Integer，在这种情况下，切出来的图片的形状是正方形。
    # transforms.Grayscale(), # 灰度话
    transforms.ToTensor(),
                                      # transforms.ToTensor()函数的作用是将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
                                      # 先由HWC转置为CHW格式；
                                      # 再转为float类型；
                                      # 最后，每个像素除以255,从0-255变换到0-1之间。
    transforms.Normalize(mean = [0.458,0.456,0.406],std = [0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    # transforms.Grayscale(), # 灰度话
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.458,0.456,0.406],std = [0.229,0.224,0.225])
])




#根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data=MyDataset(root,'train_data.txt', transform=train_transform)
test_data=MyDataset(root,'test_data.txt', transform=test_transform)

# 数据集类别，批量参数
classes = ('0', '1')
batch_size = 1


# 数据加载器，调用datasets
"""
torch.utils.data.DataLoader：数据加载器，自定义dataset封装在其中。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
参数：

dataset (Dataset) – 加载数据的数据集。
batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
collate_fn (callable, optional) –
pin_memory (bool, optional) –
drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
"""
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


""" 
  torch.flatten(x, 1)：
    input: 一个 tensor，即要被“推平”的 tensor。
    start_dim: “推平”的起始维度。
    end_dim: “推平”的结束维度。
    本函数只有start_dim=1，因此将维度1及以后的所有维度展品
"""

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  #第一层卷积层
#         x = self.pool(F.relu(self.conv2(x)))  #第二层卷积层
#         # x=torch.IntTensor(16,5,5)
#         # x = torch.flatten(x,1) # flatten all dimensions except batch
#         # x = torch.tensor(16 * 5 * 5)
#         # x = torch.flatten(torch.tensor(16 * 5 * 5), 1)
#         # x = x.view(-1,16 * 5 * 5)

#         # x = x.view(x.size(0), 35820144)
#         x = x.view(-1, 16 * 5* 5)

#         x = F.relu(self.fc1(x)) #全连接层
#         x = F.relu(self.fc2(x)) #全连接层
#         x = self.fc3(x) #全连接层
#         return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.classifer = nn.Sequential(
            nn.Linear(4*4*16,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
        )
    def forward(self,x):
        feature = self.feature(x)
        classifer = self.classifer(feature.view(feature.shape[0],-1))
        return classifer

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        # self.fc1 = nn.Linear(54*54*16,120)
        self.fc1 = nn.Linear(16*54*54,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,16*54*54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim = 1)

class Net2(nn.Module):                 # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):   
        super(Net2, self).__init__()   # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)       # 添加第一个卷积层,调用了nn里面的Conv2d（）
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)        # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)      # 同样是卷积层
       #self.fc1 = nn.Linear(16 * 5 * 5, 120) # 接着三个全连接层 ，原代码，错误的代码
        self.fc1 = nn.Linear(16 * 53 * 53, 120) # 接着三个全连接层 修改后的正确的代码
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
  
    def forward(self, x):                  # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
                                           # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5) #原代码，错误的代码
        x = x.view(-1, 16 * 53 * 53) #修改后的正确的代码
        #179776= 16*53*53*4 解读 4 是4批
        

        #x = x.view(-1, self.num_flat_features(x)) #为什么我要用这个？
        
        #x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
                                    #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
                                    #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
                                    #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
                                    # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net()
# net = LeNet()
net = CNNModel()
# net = Net2()
net = net.cuda()
# gpus = [0]   #使用哪几个GPU进行训练，这里选择0号GPU
# cuda_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')##################################
# if(cuda_gpu):
#     net = torch.nn.DataParallel(net, device_ids=gpus).cuda()   #将模型转为cuda类型


criterion = nn.CrossEntropyLoss() #交叉熵损失函数
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #动量的SGD优化器
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)


# for epoch in range(10):  # loop over the dataset multiple times

#     running_loss = 0.0  #声明损失
#     for i, data in enumerate(trainloader, 0): #从数据加载器中提取数据及索引
#         # get the inputs; data is a list of [inputs, labels]
#         # data=data.to(device)
#         inputs, labels = data # 数据、标签
#         # inputs=inputs.to(device)#######################################
#         # labels=labels.to(device)#####################################

#         # zero the parameter gradients将参数梯度归零
#         # https://blog.csdn.net/scut_salmon/article/details/82414730
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         # if(cuda_gpu):
#         #     net = torch.nn.DataParallel(net, device_ids=gpus).cuda()   #将模型转为cuda类型
#         # print(inputs.is_cuda)   # 查看输入是否在cuda上
#         # print(next(net.parameters()).device)    # 查看模型是否在cuda上
#         outputs = net(inputs)# 网络预测########################################
#         # print(outputs.is_cuda)
#         loss = criterion(outputs, labels) #计算损失
#         loss.backward() #反向传播
#         optimizer.step()  #在反向传播后调用该指令起到更新网络参数的作用

#         # print statistics显示结果
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

import time
import math
start = time.time()
epochs = 10
train_losses = []
train_correct = []
test_losses = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b,(X_train,Y_train) in enumerate(trainloader):
        b+=1
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        Y_pred = net(X_train)
        loss = criterion(Y_pred,Y_train)
        predictions = torch.max(Y_pred.data,1)[1]
        trn_corr+= (predictions == Y_train).sum()
        if b%125==0:
            print(f'Epoch {i+1} batch:{b} [{b*10}/{18750}] loss:{loss.item():.2f} accuracy:{(trn_corr.item()*100)/(10*b):.2f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(loss)
    train_correct.append(trn_corr)
    with torch.no_grad():
        for(X_test,Y_test) in testloader:
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
            Y_val = net(X_test)
            predictions = torch.max(Y_val.data,1)[1]
            tst_corr+= (predictions == Y_test).sum()
        loss = criterion(Y_val,Y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
end = time.time()
print(f'Train Duration : {(end-start)//60} minutes {math.ceil((end-start)%60)} seconds')



print('--------------------------------------------------------------------------------------')
plt.plot(train_losses,label = 'Train Losses')
plt.plot(test_losses,label = 'Test Losses')
plt.legend()
print('--------------------------------------------------------------------------------------')
plt.plot([i for i in range(1,11)],[100*i/18743 for i in train_correct],label = 'Training Accuracy')
plt.plot([i for i in range(1,11)],[100*i/6251 for i in test_correct],label = 'Testing Accuracy')
plt.legend()

PATH = './run/train/mode1.pth'
torch.save(net.state_dict(), PATH)