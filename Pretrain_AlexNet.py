# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:08:10 2018

@author: norman
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image

def is_cuda():
    if torch.cuda.is_available():
        return True
    else:
        return False


def getAlexNetMdoel():
    alexNetmodel = models.alexnet(True)
    if is_cuda():
        alexNetmodel.cuda()
    return alexNetmodel


def classify(net_model, img):
    net_model.eval()

    pre_tensor = transforms.ToTensor()(img)
    pre_tensor = pre_tensor.resize_(1, 3, 224, 224).cuda()

    result = net_model(Variable(pre_tensor))
    result_cpu = result.data.cpu().numpy()
    idx = np.argmax(result_cpu)
    print(idx)
    return idx


if __name__=='__main__':
    net_model = getAlexNetMdoel()
    # img = cv2.imread('E:/caffe-master-GPU/examples/images/cat.jpg')
    # img = Image.open('E:/caffe-master-GPU/examples/images/cat.jpg')
    # idx = classify(net_model, img)
    labels_file = "C:/CodeFolder/DeepLearning/synset_words.txt"
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    # print('output label:', labels[idx])
    net_model = getAlexNetMdoel()
    img = cv2.imread('C:/CodeFolder/DeepLearning/cat.jpg')
    # img = Image.open('E:/caffe-master-GPU/examples/images/cat.jpg')
    # idx = classify(net_model, img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    pre_tensor = transforms.ToTensor()(img)
    pre_tensor = pre_tensor.resize_(1, 3, 224, 224)
    image = Variable(pre_tensor.cuda())
    out = net_model(image)
    _, predicted = torch.max(out.data, 1)
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    print('output label:', labels[predicted], predicted)