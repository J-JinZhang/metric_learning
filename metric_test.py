import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
import random
import os

import matplotlib.pyplot as plt
from inception_emb import Img_Inception
from inception_net import Froth_Inception


root = './data'
"""imgs = ["1_20170903231055_EP_1_5.jpg",
                    "1_20170903231055_PLUS_3_6.jpg",
                    "2_20170830120026_EP_2_2.jpg",
                    "2_20170830120026_EP_2_4.jpg",
                    "3_20170830095017_EP_1_2.jpg",
                    "3_20170830095017_EP_1_6.jpg",
                    "3_20170831010642_EP_2_1.jpg",
                    "3_20170831010642_EP_4_1.jpg",
                    "4_20170830101351_EP_6_2.jpg",
                    "4_20170830101351_PLUS_1_6.jpg",
                    "5_20170830125551_EP_1_5.jpg",
                    "5_20170830125551_EP_2_5.jpg"]"""
imgs = ["1_20170903231055_EP_1_5.jpg",
        "1_20170903231055_PLUS_3_6.jpg",
        "2_20170830120026_EP_2_2.jpg",
        "2_20170830120026_EP_2_4.jpg",
        "3_20170830095017_EP_1_2.jpg",
        "3_20170830095017_EP_1_6.jpg",
        "4_20170830101351_EP_6_2.jpg",
        "4_20170830101351_PLUS_1_6.jpg"]
        
normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491],
                                                          std=[0.1833, 0.1916, 0.2061])
transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(300),
                                                              torchvision.transforms.ToTensor(),
                                                              normalize])
criterion = torch.nn.MSELoss()
COS_criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

metric_model = Img_Inception()
save_file = os.path.join('./saved_models', 'MetricNet_epoch_300.pth')
metric_model.load_state_dict(torch.load(save_file))

froth_model = Froth_Inception()
save_file = os.path.join('./saved_models', 'XRF_InceptionNet_epoch_300.pth')
froth_model.load_state_dict(torch.load(save_file))
feature_model = froth_model.features


with torch.no_grad():
    for i in list([0,2,4,6]):#range(len(imgs)-1):
        metric_model.eval()
        img_i_path = os.path.join(root, imgs[i])
        img_i = Image.open(img_i_path).convert("RGB")
        img_i = transforms(img_i).float()
        img_i_mapping = metric_model(img_i.unsqueeze(0))
        img_i_feature = feature_model(img_i.unsqueeze(0))
        for  j in range(len(imgs)):
            img_j_path = os.path.join(root, imgs[j])
            img_j = Image.open(img_j_path).convert("RGB")
            img_j = transforms(img_j).float()
            img_j_mapping = metric_model(img_j.unsqueeze(0))
            img_j_feature = feature_model(img_j.unsqueeze(0))
            dist_metric = criterion(img_i_mapping, img_j_mapping)
            print(f"img_i_feature: {img_i_feature.size()}")
            dist_feature = COS_criterion(img_i_feature.view(1,-1), img_j_feature.view(1,-1))
            dist_EU = criterion(img_i.view(-1), img_j.view(-1))
            print(f"Dist between {i} and {j} is {dist_metric} and {dist_feature} and {dist_EU}")