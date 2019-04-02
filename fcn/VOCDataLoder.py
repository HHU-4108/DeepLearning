import os
import collections
import numpy as np
import PIL.Image
# import scipy.io
import torch
from torch.utils import data

class VOCLoader(data.Dataset):
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

    def __init__(self, data_folder, transform = None):
        self.transform = transform
        self.data_folder = data_folder
        self.files = collections.defaultdict(list)

        with open(os.path.join(self.data_folder, 'ImageSets/Segmentation/train.txt')) as f:
            img_name = f.readline().strip()
            img_file = os.path.join(self.data_folder, "JPEGImages/%s.jpg"%img_name)
            label_file = os.path.join(self.data_folder, 'SegmentationClass/%s.png'%img_name)
            self.files.apend({'img': img_file, 'label': label_file})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data_path = self.files[item]

        img = PIL.Image.open(data_path['img'])
        img = np.array(img, dtype=np.uint8)

        label = PIL.Image.open(data_path['label'])
        label = np.array(label, dtype=np.int32)

        return img, label
