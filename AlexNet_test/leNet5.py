# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:08:10 2018

@author: norman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision


trainSet = torchvision.datasets.MNIST(root='./mnist/', # dataset存储路径
             train=True, # True表示是train训练集，False表示test测试集
             transform=torchvision.transforms.ToTensor(), # 将原数据规范化到（0,1）区间
             download= False)
testSet = torchvision.datasets.MNIST(root='./mnist/', # dataset存储路径
             train=False, # True表示是train训练集，False表示test测试集
             transform=torchvision.transforms.ToTensor(), # 将原数据规范化到（0,1）区间
             download= False)

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=4, 
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testSet, batch_size=4, 
                                          shuffle=False, num_workers=2)


class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*16, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)# 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)# 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
 
    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到 x。
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(x.size(0), -1) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x

 
net = Net()

if torch.cuda.is_available():
    net.cuda()#将所有的模型参数移动到GPU上
criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

if __name__ == '__main__':
    print("start training>>>>>>>>>>>>>>>>>>>")
    for epoch in range(2): # 遍历数据集两次

        running_loss = 0.0
        #enumerate(sequence, [start=0])，i序号，data是数据
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data   #data的结构是：[4x1x28x28的张量,长度4的张量]

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
            if torch.cuda.is_available():
                datas, target = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad() #将参数的grad值初始化为0

            # forward + backward + optimize
            outputs = net(datas)
            loss = criterion(outputs, target) #将output和labels使用叉熵计算损失
            loss.backward() #反向传播
            optimizer.step() #用SGD更新参数

            # 每2000批数据打印一次平均loss值
            running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
            if i % 2000 == 1999: # 每2000批打印一次
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images).cuda())
        #print outputs.data
        _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        total += labels.size(0)
        correct += (predicted == Variable(labels).cuda()).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
