import os
import numpy as np
import PIL.Image
# import scipy.io
import torch
from torch.utils import data
from torchvision import  transforms

class VOCLoader(data.Dataset):
    trsf = transforms.Compose([transforms.ToTensor()])
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_folder, transform=True):
        self._transform = transform
        self.data_folder = data_folder
        self.img_files = []
        self.lbl_files = []

        with open(os.path.join(self.data_folder, 'ImageSets/Segmentation/train.txt')) as f:
            img_name = f.readline().strip()
            img_file = os.path.join(self.data_folder, "JPEGImages/%s.jpg"%img_name)
            label_file = os.path.join(self.data_folder, 'SegmentationClass/%s.png'%img_name)
            self.img_files.append(img_file)
            self.lbl_files.append(label_file)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        # data_path = self.files[item]

        img = PIL.Image.open(self.img_files[item])
        img = np.array(img, dtype=np.float64)
        img -= self.mean_bgr
        print("img shape")
        print(img.shape)
        label = PIL.Image.open(self.lbl_files[item])
        label = np.array(label, dtype=np.int32)
        label[label == 255] = -1
        if self._transform:
            img = self.trsf(img)
        return img, torch.from_numpy(label)

    # def transform(self, img, lbl):
    #     img = img[:, :, ::-1]  # RGB -> BGR
    #     img = img.astype(np.float64)
    #     img -= self.mean_bgr
    #     img = img.transpose(2, 0, 1)
    #     img = torch.from_numpy(img).float()
    #     lbl = torch.from_numpy(lbl).long()
    #     print(img.shape)
    #     return img, lbl
    #
    # def untransform(self, img, lbl):
    #     img = img.numpy()
    #     img = img.transpose(1, 2, 0)
    #     img += self.mean_bgr
    #     img = img.astype(np.uint8)
    #     img = img[:, :, ::-1]
    #     lbl = lbl.numpy()
    #     return img, lbl
