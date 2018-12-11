#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:11:08 2018

@author: qzane
"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import tqdm #progress bar 
import time
import os
import copy

from trainer_v2 import train_model
from dataset import DataSet
from model import initialize_model

from dataPath import SAVE_PATH, TRAIN_PATH, VAL_PATH


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
    
# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras


def f1_loss(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred)
    epsilon = 1e-07
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = torch.sum(y_true*y_pred, dim=0)
    tn = torch.sum((1-y_true)*(1-y_pred), dim=0)
    fp = torch.sum((1-y_true)*y_pred, dim=0)
    fn = torch.sum(y_true*(1-y_pred), dim=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2*p*r / (p+r+epsilon)
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-f1.mean()

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

data_dir = {'train':TRAIN_PATH,
            'val':VAL_PATH}

model_name = 'densenet161v5'

num_classes = 28

batch_size = 128

num_epochs = 50

feature_extract = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft, input_size = initialize_model(model_name, num_classes, 
                                        feature_extract, use_pretrained=True)
#print(model_ft)

# Create training and validation datasets
image_datasets = {x: DataSet(data_dir[x], input_size, num_classes, x=='train') for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
                                                   shuffle=True, num_workers=16) for x in ['train', 'val']}

for x in ['train', 'val']:
    print('dataset size', x, len(image_datasets[x]))
    print('loader size', x, len(dataloaders_dict[x]))
    
    
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


#optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update, lr=0.003)


#criterion = nn.BCELoss()
criterion = f1_loss#FocalLoss()

# Train and evaluate

model_ft, hist = train_model(device, model_ft, dataloaders_dict, 
                             criterion, optimizer_ft, num_classes,
                             num_epochs=num_epochs, 
                             is_inception=(model_name=="inception"), 
                             model_name=model_name,
                             save_path=SAVE_PATH)











