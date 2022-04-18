# -*- coding: unicode_escape -*-
"cxking"
# ����torchģ��
import torch
#���ݴ���ģ��
import torchvision
import torchvision.transforms as transforms
#ģ���Ż�ģ��
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# ����ģ��
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 #�������õľ���
import os
import warnings
warnings.filterwarnings('ignore')

# ����·��
root='/home/cxking/datasets/line_data_new/'
# ���ݴ���class
class MyDataset(torch.utils.data.Dataset): #�����Լ����ࣺMyDataset,������Ǽ̳е�torch.utils.data.Dataset
    def __init__(self,root, datatxt, transform=None, target_transform=None): #��ʼ��һЩ��Ҫ����Ĳ���
        print(os.path.join(root,datatxt))
        fh = open(root + datatxt, 'r') #���մ����·����txt�ı�������������ı�������ȡ����
        imgs = []                      #����һ����Ϊimg�Ŀ��б�һ�������װ����
        for line in fh:                #����ѭ��txt�ı��е�����
            line = line.rstrip()       # ɾ�� ����string �ַ���ĩβ��ָ���ַ��������������ϸ�����Լ���ѯpython
            words = line.split()   #ͨ��ָ���ָ������ַ���������Ƭ��Ĭ��Ϊ���еĿ��ַ��������ո񡢻��С��Ʊ����
            imgs.append((words[0],int(words[1]))) #��txt������ݶ���imgs�б��棬������words��Ҫ��txt���ݶ���
                                        # ����Ȼ�������ҸղŽ�ͼ��ʾtxt�����ݣ�words[0]��ͼƬ��Ϣ��words[1]��lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    #���ڰ���������ȡÿ��Ԫ�صľ�������
    def __getitem__(self, index):    
        fn, label = self.imgs[index] #fn��ͼƬpath #fn��label�ֱ���imgs[index]Ҳ���Ǹղ�ÿ����word[0]��word[1]����Ϣ
        img = Image.open(root+fn).convert('RGB') #����path����ͼƬfrom PIL import Image # ����·����ȡͼƬ
 
        if self.transform is not None:
            img = self.transform(img) #�Ƿ����transform
        return img,label  #return�ܹؼ���return����Щ���ݣ���ô������ѵ��ʱѭ����ȡÿ��batchʱ�����ܻ����Щ����
 
    def __len__(self): #�������Ҳ����Ҫд�������ص������ݼ��ĳ��ȣ�Ҳ���Ƕ�����ͼƬ��Ҫ��loader�ĳ���������
        return len(self.imgs)
 


train_transform = transforms.Compose([  #�������������
    transforms.RandomRotation(10),  #��ָ���ĽǶ�ѡװͼƬ����-10��10����ת�Ƕȷ�Χ
    transforms.RandomHorizontalFlip(),  # ���ˮƽ��ת������PIL.Image,����Ϊ0.5������һ��ĸ��ʷ�ת��һ��ĸ��ʲ���ת��
    transforms.Resize(224),
    transforms.CenterCrop(224), #��������PIL.Image���������и�õ�������size��size������tuple��(target_height, target_width)��sizeҲ������һ��Integer������������£��г�����ͼƬ����״�������Ρ�
    # transforms.Grayscale(), # �ҶȻ�
    transforms.ToTensor(),
                                      # transforms.ToTensor()�����������ǽ�ԭʼ��PILImage��ʽ����numpy.array��ʽ�����ݸ�ʽ��Ϊ�ɱ�pytorch���ٴ�����������͡�
                                      # ����HWCת��ΪCHW��ʽ��
                                      # ��תΪfloat���ͣ�
                                      # ���ÿ�����س���255,��0-255�任��0-1֮�䡣
    transforms.Normalize(mean = [0.458,0.456,0.406],std = [0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    # transforms.Grayscale(), # �ҶȻ�
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.458,0.456,0.406],std = [0.229,0.224,0.225])
])




#�����Լ�������Ǹ���MyDataset���������ݼ���ע�������ݼ���������loader������
train_data=MyDataset(root,'train_data.txt', transform=train_transform)
test_data=MyDataset(root,'test_data.txt', transform=test_transform)

# ���ݼ������������
classes = ('0', '1')
batch_size = 1


# ���ݼ�����������datasets
"""
torch.utils.data.DataLoader�����ݼ��������Զ���dataset��װ�����С�������ݼ��Ͳ��������������ݼ����ṩ�����̻����̵�������
������

dataset (Dataset) �C �������ݵ����ݼ���
batch_size (int, optional) �C ÿ��batch���ض��ٸ�����(Ĭ��: 1)��
shuffle (bool, optional) �C ����ΪTrueʱ����ÿ��epoch���´�������(Ĭ��: False).
sampler (Sampler, optional) �C ��������ݼ�����ȡ�����Ĳ��ԡ����ָ���������shuffle������
num_workers (int, optional) �C �ö��ٸ��ӽ��̼������ݡ�0��ʾ���ݽ����������м���(Ĭ��: 0)
collate_fn (callable, optional) �C
pin_memory (bool, optional) �C
drop_last (bool, optional) �C ������ݼ���С���ܱ�batch size������������ΪTrue���ɾ�����һ����������batch�������ΪFalse�������ݼ��Ĵ�С���ܱ�batch size�����������һ��batch����С��(Ĭ��: False)
"""
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


""" 
  torch.flatten(x, 1)��
    input: һ�� tensor����Ҫ������ƽ���� tensor��
    start_dim: ����ƽ������ʼά�ȡ�
    end_dim: ����ƽ���Ľ���ά�ȡ�
    ������ֻ��start_dim=1����˽�ά��1���Ժ������ά��չƷ
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
#         x = self.pool(F.relu(self.conv1(x)))  #��һ������
#         x = self.pool(F.relu(self.conv2(x)))  #�ڶ�������
#         # x=torch.IntTensor(16,5,5)
#         # x = torch.flatten(x,1) # flatten all dimensions except batch
#         # x = torch.tensor(16 * 5 * 5)
#         # x = torch.flatten(torch.tensor(16 * 5 * 5), 1)
#         # x = x.view(-1,16 * 5 * 5)

#         # x = x.view(x.size(0), 35820144)
#         x = x.view(-1, 16 * 5* 5)

#         x = F.relu(self.fc1(x)) #ȫ���Ӳ�
#         x = F.relu(self.fc2(x)) #ȫ���Ӳ�
#         x = self.fc3(x) #ȫ���Ӳ�
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

class Net2(nn.Module):                 # ���Ƕ�������ʱһ���Ǽ̳е�torch.nn.Module�����µ�����
    def __init__(self):   
        super(Net2, self).__init__()   # �ڶ������ж���python��̳еĻ�������,��д��Ӧ����python2.7�ļ̳и�ʽ,��python3��д�������Ҳ����
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)       # ��ӵ�һ�������,������nn�����Conv2d����
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)        # ���ػ���
        self.conv2 = nn.Conv2d(6, 16, 5)      # ͬ���Ǿ����
       #self.fc1 = nn.Linear(16 * 5 * 5, 120) # ��������ȫ���Ӳ� ��ԭ���룬����Ĵ���
        self.fc1 = nn.Linear(16 * 53 * 53, 120) # ��������ȫ���Ӳ� �޸ĺ����ȷ�Ĵ���
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
  
    def forward(self, x):                  # ���ﶨ��ǰ�򴫲��ķ�����Ϊʲôû�ж��巴�򴫲��ķ����أ�����ʵ���漰��torch.autogradģ���ˣ�
                                           # ��˵ʵ���ⲿ�����綨��Ĳ��ֻ�û���õ�autograd��֪ʶ�����Ժ����������ٽ�
        x = self.pool(F.relu(self.conv1(x)))  # F��torch.nn.functional�ı��������������relu���� F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5) #ԭ���룬����Ĵ���
        x = x.view(-1, 16 * 53 * 53) #�޸ĺ����ȷ�Ĵ���
        #179776= 16*53*53*4 ��� 4 ��4��
        

        #x = x.view(-1, self.num_flat_features(x)) #Ϊʲô��Ҫ�������
        
        #x = x.view(-1, 16 * 5 * 5)  # .view( )��һ��tensor�ķ�����ʹ��tensor�ı�size����Ԫ�ص������ǲ���ġ�
                                    #  ��һ������-1��˵�����������һ������ȷ���� ���������Ԫ������һ��������£�ȷ����������ȷ��������
                                    #  ��ôΪʲô����ֻ�������������������أ���Ϊ���Ͼ�Ҫ����ȫ���Ӳ��ˣ���ȫ���Ӳ�˵���˾��Ǿ���˷���
                                    #  ��ᷢ�ֵ�һ��ȫ���Ӳ���ײ�����16*5*5������Ҫ��֤�ܹ���ˣ��ھ���˷�֮ǰ��Ҫ��x������ȷ��size
                                    # �����Tensor�����ο�Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net()
# net = LeNet()
net = CNNModel()
# net = Net2()
net = net.cuda()
# gpus = [0]   #ʹ���ļ���GPU����ѵ��������ѡ��0��GPU
# cuda_gpu = torch.cuda.is_available()   #�ж�GPU�Ƿ���ڿ���
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')##################################
# if(cuda_gpu):
#     net = torch.nn.DataParallel(net, device_ids=gpus).cuda()   #��ģ��תΪcuda����


criterion = nn.CrossEntropyLoss() #��������ʧ����
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #������SGD�Ż���
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)


# for epoch in range(10):  # loop over the dataset multiple times

#     running_loss = 0.0  #������ʧ
#     for i, data in enumerate(trainloader, 0): #�����ݼ���������ȡ���ݼ�����
#         # get the inputs; data is a list of [inputs, labels]
#         # data=data.to(device)
#         inputs, labels = data # ���ݡ���ǩ
#         # inputs=inputs.to(device)#######################################
#         # labels=labels.to(device)#####################################

#         # zero the parameter gradients�������ݶȹ���
#         # https://blog.csdn.net/scut_salmon/article/details/82414730
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         # if(cuda_gpu):
#         #     net = torch.nn.DataParallel(net, device_ids=gpus).cuda()   #��ģ��תΪcuda����
#         # print(inputs.is_cuda)   # �鿴�����Ƿ���cuda��
#         # print(next(net.parameters()).device)    # �鿴ģ���Ƿ���cuda��
#         outputs = net(inputs)# ����Ԥ��########################################
#         # print(outputs.is_cuda)
#         loss = criterion(outputs, labels) #������ʧ
#         loss.backward() #���򴫲�
#         optimizer.step()  #�ڷ��򴫲�����ø�ָ���𵽸����������������

#         # print statistics��ʾ���
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