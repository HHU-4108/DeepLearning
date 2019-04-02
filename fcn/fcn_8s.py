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
        x = x[:, :, 2:2 + upscore2.size()[2], 2:2 + upscore2.size()[3]]
        x = x + upscore2
        x = self.upscore_pool4(x)
        upscore_pool4 = x

        x = self.score_pool3(pool3)
        x = x[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        x = x + upscore_pool4

        x = self.upscore8(x)
        x = x[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return x

    def copy_weight(self, vgg16):
        features = [
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.conv5_1,
            self.conv5_2,
            self.conv5_3
        ]

        for l1, l2 in zip(features, vgg16.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                    l1.weight.data.copy_(l2.weight.data)
                    l1.bias.data.copy_(l2.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = getattr(self, name)
            l2 = vgg16.classifier[i]
            l1.weight.data.copy_(l2.weight.data.view(l1.weight.size()))
            l1.bias.data.copy_(l2.bias.data.view(l1.bias.size()))


model = load_pretrained_net()
print(model)
