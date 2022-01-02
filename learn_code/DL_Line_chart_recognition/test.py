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
import time
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



# # չʾ���ݼ�

# # functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    time.sleep(2)
    plt.close()
    # print ('it is ok')


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



#����
# ��ȡ��������
dataiter = iter(testloader)
images, labels = dataiter.next()
print('labels:',labels.shape)
print('img:',images)
# print imagesչʾ��ȷgt
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(1)))


# ��������Ȩ��
# net = Net()
net = CNNModel()
net.load_state_dict(torch.load('./run/train/mode1.pth'))

outputs = net(images)

# ȡ�����ʵ����Ϊ���
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(1)))

#�����������磬������ȷ��
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network 
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 100 test images: %d %%' % (
    100 * correct / total))
"""
/home/cxking/datasets/line_data_new/train_data.txt
/home/cxking/datasets/line_data_new/test_data.txt
num_of_trainData: 84
num_of_testData: 21
labels: torch.Size([1])
img: tensor([[[[1.8188, 1.8188, 1.8188,  ..., 1.8188, 1.8188, 1.8188],
          [2.2812, 2.2812, 2.2812,  ..., 2.2812, 2.2812, 2.2812],
          [2.3668, 2.3668, 2.3668,  ..., 2.3668, 2.3668, 2.3668],
          ...,
          [2.3668, 2.3668, 2.3668,  ..., 2.3668, 2.3668, 2.3668],
          [2.2812, 2.2812, 2.2812,  ..., 2.2812, 2.2812, 2.2812],
          [1.8188, 1.8188, 1.8188,  ..., 1.8188, 1.8188, 1.8188]],

         [[1.8683, 1.8683, 1.8683,  ..., 1.8683, 1.8683, 1.8683],
          [2.3410, 2.3410, 2.3410,  ..., 2.3410, 2.3410, 2.3410],
          [2.4286, 2.4286, 2.4286,  ..., 2.4286, 2.4286, 2.4286],
          ...,
          [2.4286, 2.4286, 2.4286,  ..., 2.4286, 2.4286, 2.4286],
          [2.3410, 2.3410, 2.3410,  ..., 2.3410, 2.3410, 2.3410],
          [1.8683, 1.8683, 1.8683,  ..., 1.8683, 1.8683, 1.8683]],

         [[2.0823, 2.0823, 2.0823,  ..., 2.0823, 2.0823, 2.0823],
          [2.5529, 2.5529, 2.5529,  ..., 2.5529, 2.5529, 2.5529],
          [2.6400, 2.6400, 2.6400,  ..., 2.6400, 2.6400, 2.6400],
          ...,
          [2.6400, 2.6400, 2.6400,  ..., 2.6400, 2.6400, 2.6400],
          [2.5529, 2.5529, 2.5529,  ..., 2.5529, 2.5529, 2.5529],
          [2.0823, 2.0823, 2.0823,  ..., 2.0823, 2.0823, 2.0823]]]])
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
GroundTruth:      0
Predicted:      1
Accuracy of the network on the 100 test images: 85 %

"""

# �鿴ÿ������׼ȷ��
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data    
        outputs = net(images)    
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

  
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                   accuracy))
"""
(torch180) cxking@cxking:~/project/Line_chart_recognition$ python test.py 
/home/cxking/datasets/line_data_new/train_data.txt
/home/cxking/datasets/line_data_new/test_data.txt
num_of_trainData: 84
num_of_testData: 21
labels: torch.Size([1])
img: tensor([[[[1.8188, 1.8188, 1.8188,  ..., 1.8188, 1.8188, 1.8188],
          [2.2812, 2.2812, 2.2812,  ..., 2.2812, 2.2812, 2.2812],
          [2.3668, 2.3668, 2.3668,  ..., 2.3668, 2.3668, 2.3668],
          ...,
          [2.3668, 2.3668, 2.3668,  ..., 2.3668, 2.3668, 2.3668],
          [2.2812, 2.2812, 2.2812,  ..., 2.2812, 2.2812, 2.2812],
          [1.8188, 1.8188, 1.8188,  ..., 1.8188, 1.8188, 1.8188]],

         [[1.8683, 1.8683, 1.8683,  ..., 1.8683, 1.8683, 1.8683],
          [2.3410, 2.3410, 2.3410,  ..., 2.3410, 2.3410, 2.3410],
          [2.4286, 2.4286, 2.4286,  ..., 2.4286, 2.4286, 2.4286],
          ...,
          [2.4286, 2.4286, 2.4286,  ..., 2.4286, 2.4286, 2.4286],
          [2.3410, 2.3410, 2.3410,  ..., 2.3410, 2.3410, 2.3410],
          [1.8683, 1.8683, 1.8683,  ..., 1.8683, 1.8683, 1.8683]],

         [[2.0823, 2.0823, 2.0823,  ..., 2.0823, 2.0823, 2.0823],
          [2.5529, 2.5529, 2.5529,  ..., 2.5529, 2.5529, 2.5529],
          [2.6400, 2.6400, 2.6400,  ..., 2.6400, 2.6400, 2.6400],
          ...,
          [2.6400, 2.6400, 2.6400,  ..., 2.6400, 2.6400, 2.6400],
          [2.5529, 2.5529, 2.5529,  ..., 2.5529, 2.5529, 2.5529],
          [2.0823, 2.0823, 2.0823,  ..., 2.0823, 2.0823, 2.0823]]]])
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
GroundTruth:      0
Predicted:      1
Accuracy for class 0     is: 40.0 %
Accuracy for class 1     is: 100.0 %
"""
