# 1. Import library
import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet18, resnet34
from collections import OrderedDict


# 2. Model class
class HydraNetModified(nn.Module):
    def __init__(self, net):
        super(HydraNetModified, self).__init__()
        self.net = net
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 1))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 1))]))
        self.net.fc3 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 5))]))
        
    def forward(self, x):
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        race_head = self.net.fc3(self.net(x))
        return age_head, gender_head, race_head


# 3. Load model function
def load_model(model_path=None):

    net = resnet34(weights='IMAGENET1K_V1')
    model = HydraNetModified(net)

    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


# # 4. Try loading model
# model = load_model()
# ## Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# print("MODEL")
# print(model)
# print(" ")
# print("DEVICE: ", device)