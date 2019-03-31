import torch.nn as nn
from torchvision import models


def load_pretrained_net():
    vgg16_model = models.vgg16(pretrained=True)
    my_model = vgg16_model.features
    return vgg16_model


class FCN_8s(nn.Module):
    def __init__(self, pretrained_net, num_calss):
        self.classify_sequential = pretrained_net
        # first sequential
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # second sequential
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # three sequential
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # fourth sequential
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # five sequential
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # full conv layers
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_calss, 1)
        self.score_pool4 = nn.Conv2d(512, num_calss, 1)
        self.score_pool3 = nn.Conv2d(256, num_calss, 1)

        self.upscore2 = nn.ConvTranspose2d(num_calss, num_calss, 4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_calss, num_calss, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_calss, num_calss, kernel_size=16, stride=8, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        pool3 = self.pool3(x)

        x = self.relu(self.conv4_1(pool3))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        pool4 = self.pool4(x)

        x = self.relu(self.conv5_1(pool4))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool5(x)

        x = self.relu(self.fc6(x))
        x = self.drop6(x)

        x = self.relu(self.fc7(x))
        x = self.drop7(x)

        x = self.score_fr(x)
        x = self.upscore2(x)
        upscore2 = x

        x = self.score_pool4(pool4)
        x = 


model = load_pretrained_net()
print(model)