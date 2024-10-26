import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple
from labels import GTA5Labels_TaskCV2017


class GTA5(Dataset):

    label_map = GTA5Labels_TaskCV2017()

    def __init__(self, root_dir, split='train', transform_img=None, transform_lab=None, transform_paired=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test' split.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform_img = transform_img
        self.transform_lab = transform_lab
        self.transform_paired=transform_paired
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.label_dir = os.path.join(self.root_dir, 'labels')
        self.images = os.listdir(self.image_dir)

        self.dataset = []
        for sample in self.images:
            image = os.path.join(self.image_dir, sample)
            label = os.path.join(self.label_dir, sample)
            self.dataset.append((image, label))

    
    @classmethod
    def encode(cls, color_img):
        return cls._encode(color_img, label_map=cls.label_map.list_)

    @staticmethod
    def _encode(color_img, label_map):
        lbl = np.zeros(color_img.shape[:2], dtype=np.int32)
        for label in label_map:
            mask = np.all(color_img == label.color, axis=-1)
            lbl[mask] = label.ID
        return lbl

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        lbl = lbl.numpy()
        color_lbl = np.full((*lbl.shape, 3), 255)
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl

  
    

    def __getitem__(self, idx):
        img_path, lblID_path = self.dataset[idx]

        image = Image.open(img_path).convert('RGB')
        labelID = Image.open(lblID_path).convert('RGB')

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_paired:
            image, labelID = self.transform_paired(image, labelID)
        if self.transform_lab:
            labelID = self.transform_lab(labelID)
        
        
        labelID = np.array(labelID)
        labelID = GTA5.encode(labelID)

        return image, labelID

    def __len__(self):
        return len(self.dataset)