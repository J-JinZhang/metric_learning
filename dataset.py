"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.17
"""

import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
import random
import os

import matplotlib.pyplot as plt

#定义一个数据集
class Data4MetricLearn(Dataset):
    def __init__(self):
        csv_file = 'AllImageClip.csv'
        self.root = '/media/neuralits/Data_SSD/FrothData/Data4FrothGrade'
        self.df=pd.read_csv(csv_file)
        im_clip = self.df.iloc[:,0].values
        
        index = np.random.RandomState(seed=42).permutation(len(self.df)) #np.random.permutation(len(self.df))
        self.im_clip = im_clip[index]
        
        transform = None
        if transform is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491],
                                                          std=[0.1833, 0.1916, 0.2061])
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(300),
                                                              torchvision.transforms.ToTensor(),
                                                              normalize])
        
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, time_idx):
        permut_idx = np.random.permutation(6)
        
        
        
        
        
        
            
        
        clip_name = "{}/clip_1".format(time_stamp)
        neg_file_name = "{}_{}.jpg".format(time_stamp, random.randint(1,6))
        neg_full_img_path = os.path.join(self.root, clip_name, neg_file_name)
        neg_img = Image.open(neg_full_img_path).convert("RGB")
        
        anchor_img = self.transforms(anchor_img).float()
        pos_img= self.transforms(pos_img).float()
        neg_img = self.transforms(neg_img).float()
        
        return anchor_img, pos_img, neg_img
