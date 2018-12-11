#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:34:00 2018

@author: qzane
"""

from __future__ import print_function 
from __future__ import division

import os,sys

import collections

import torch
import torch.utils
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

import numpy as np


class pass_through(torch.nn.Module):
    def __init__(self):
        super(pass_through, self).__init__()
    def forward(self, x):
        return x



def disp(mat):
    if type(mat) == torch.Tensor:
        mat = mat.numpy()
    if len(mat.shape) == 2:
        mat = (mat-mat.mean())/mat.max()*255
        mat = mat.astype(np.uint8)
        img = Image.fromarray(mat)
        img.show()
        
    if len(mat.shape) == 3:
        for i in range(mat.shape[0]):
            mat[i] = (mat[i]-mat[i].mean())/mat[i].max()*255
            mat[i] = mat[i].astype(np.uint8)
            img = Image.fromarray(mat[i])
            img.show()
        

def get_transforms(input_size, augmentation=True):
    if augmentation:
        return transforms.Compose([
            transforms.RandomRotation(180, resample=Image.BICUBIC),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0, 0], [255, 255, 255, 255])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0, 0], [255, 255, 255, 255])
        ])
    
def split_data(source_dir='./data/train_all', train_dir='./data/train', 
               val_dir='./data/val', classbook='./data/class.csv'):
    import os
    import random
    import shutil
    from collections import defaultdict
    
    random.seed(1994)
    
    print('source dir:', os.path.realpath(source_dir))
    print('train dir:', os.path.realpath(train_dir))
    print('val dir:', os.path.realpath(val_dir))
    
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)
        
    
    fs = os.listdir(source_dir)
    objs = defaultdict(list)
    for i in fs:
        if i.endswith('.png'):
            objs[i.split('_')[0]].append(i)
        
    print(len(objs),'length')
    lobjs = list(objs.keys())
    lobjs.sort()
    
    random.shuffle(lobjs)
    
    lall = len(objs)
    ltrain = int(lall*0.7)
    
    
    for i in range(ltrain):
        for j in objs[lobjs[i]]:
            shutil.move(os.path.join(source_dir, j), os.path.join(train_dir, j))
    for i in range(ltrain, lall):
        for j in objs[lobjs[i]]:
            shutil.move(os.path.join(source_dir, j), os.path.join(val_dir, j))
            
    shutil.copy(classbook, train_dir)
    shutil.copy(classbook, val_dir)
    
    
class DataSet(torch.utils.data.Dataset):
    
    def get_objs(self, path):
        ''' return {uid:[file1,file2,...]} '''
        objs = collections.defaultdict(lambda :[0,0,0,0])
        pos = ['red', 'green', 'blue', 'yellow']
        
        for p,_,fs in os.walk(path):
            for f in fs:
                if f.endswith('.png'):
                    uid = f.split('_')[0]
                    for i,j in enumerate(pos):
                        if j in f:
                            objs[uid][i] = os.path.join(p, f)
        for obj in objs:
            full = True
            for i in objs[obj]:
                if i == 0:
                    full = False
            if not full:
                del(objs[obj])
        return objs
                    
    def get_classes(self, class_book):
        ''' return {uid:[c1, c2,...]} '''
        if os.path.isfile(class_book):
            with open(class_book) as f:
                objs = f.readlines()
        else:
            objs = []
        res = collections.defaultdict(list)
        for obj in objs:
            try:
                uid,classes = obj.split(',')
                for i in classes.split(' '):
                    res[uid].append(int(i))
                res[uid].sort()
            except:
                pass
        return res
        
    
    def __init__(self, root=None, input_size=224, class_num=28, augmentation=True, rgby=False):        
        super(DataSet, self).__init__()
        self.objs = self.get_objs(root)
        self.classes = self.get_classes(os.path.join(root, 
                                                'class.csv'))
        self.length = len(self.objs)
        self.uids = sorted(self.objs.keys())
        
        self.input_size = input_size
        self.class_num = class_num
        
        self.trans = get_transforms(input_size, augmentation)
        self.rgby = rgby
        
        # count subjects in each class
        self.cout = collections.defaultdict(int)
        for i in self.objs:
            if i in self.classes:
                for j in self.classes[i]:
                    self.cout[j]+=1
        total = len(self.objs)
        for i in sorted(self.cout.keys()):
            print('class',i,':',
                  self.cout[i],' %.2f '%(100.0*self.cout[i]/total))
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        uid = self.uids[index]
        imR,imG,imB,imY = (Image.open(i) for i in self.objs[uid])
        if self.rgby:
            #imgs = [self.trans(Image.merge("RGB", [i,i,i])) for i in [imR,imG,imB,imY]]
            imgs = Image.merge("RGBA", [imR,imG,imB,imY])
            imgs = self.trans(imgs)
        else:
            imgs = Image.merge("RGB", [imR,imG,imB])
            imgs = self.trans(imgs)
        
        if uid in self.classes:
            target = self.classes[uid]
        else:
            target = []
        label = torch.zeros(self.class_num, dtype=torch.float)
        for i in target:
            label[i] = 1
                
        return imgs, label

             
def test_sample(img, input_size=224, augmentation=False):
    ''' img == './uid_red.png' '''
    
    pos = ['red', 'green', 'blue', 'yellow']
    objs = [img.replace('red', i) for i in pos]
    imR,imG,imB,imY = (Image.open(i) for i in objs)
    imgs = Image.merge("RGB", [imR,imG,imB])
    
    trans = get_transforms(input_size, augmentation)
    imgs = trans(imgs)
    imgs = imgs.unsqueeze(0)
    return imgs
        

def predict(path, result_file, model):
    BATCH_SIZE = 16
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    of = open(result_file, 'w')
    of.write('Id,Predicted\n')
    of.flush()
    model = model.eval().to(device)
    ds = DataSet(path, augmentation=False,rgby=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    for i,(q,w) in enumerate(loader):
        q = q.to(device)
        with torch.set_grad_enabled(False):
            vo = model(q)
        pred =  (vo>-2.9444389791664403) #  (vo>0)
        #print(pred.sum())
        for j in range(len(pred)):
            res = []
            if pred[j].sum() >= 1:
                for k in range(28):
                    if pred[j,k]:
                        res.append(str(k))
            else:
                res.append(str(vo[j].max(0)[1].item()))
                
            uid = ds.uids[i*BATCH_SIZE + j]
            #print(i, j)
            print(i*BATCH_SIZE + j,"%.2f%%"%(100.0*(i*BATCH_SIZE + j)/len(ds)), uid, res)
            of.write(uid+','+' '.join(res))
            of.write('\n')
            of.flush()
    of.close()
   

def toVec(path, result_file, model):
    BATCH_SIZE = 64
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    of = open(result_file, 'w')
    of.write('Id,Features\n')
    of.flush()
    model.classifier = pass_through()
    model = model.eval().to(device)
    ds = DataSet(path, augmentation=False,rgby=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    for i,(q,w) in enumerate(loader):
        q = q.to(device)
        with torch.set_grad_enabled(False):
            vo = model(q)
        dims = vo.size()[1]
        pred = vo
        #pred =  (vo>-2.9444389791664403) #  (vo>0)
        #print(pred.sum())
        for j in range(len(pred)):
            res = []
            if pred[j].sum() >= 1:
                for k in range(28):
                    if pred[j,k]:
                        res.append(str(k))
            else:
                res.append(str(vo[j].max(0)[1].item()))
                
            uid = ds.uids[i*BATCH_SIZE + j]
            #print(i, j)
            print(i*BATCH_SIZE + j,"%.2f%%"%(100.0*(i*BATCH_SIZE + j)/len(ds)), uid)
            of.write(uid)
            for k in range(dims):
                of.write(",%.6f"%vo[j,k])
            of.write('\n')
            of.flush()
    of.close()

def get_classes_book(class_book):
        ''' return {uid:[c1, c2,...]} '''
        if os.path.isfile(class_book):
            with open(class_book) as f:
                objs = f.readlines()
        else:
            objs = []
        res = collections.defaultdict(list)
        for obj in objs:
            try:
                uid,classes = obj.split(',')
                for i in classes.split(' '):
                    res[uid].append(int(i))
                res[uid].sort()
            except:
                pass
        return res     
import numpy as np
def f1(res1, book='./data/class.csv'):
    from sklearn.metrics import f1_score
    res1 = get_classes_book(res1)
    book = get_classes_book(book)
    keys = sorted(res1.keys())
    length = len(keys)
    r1 = np.zeros((length, 28))
    r2 = np.zeros((length, 28))
    r3 = np.zeros((len(book), 28))
    for no,key in enumerate(book):
        for i in book[key]:
            r3[no, i]=1
    
    for no,key in enumerate(keys):
        for i in res1[key]:
            r1[no, i] = 1
        for i in book[key]:
            r2[no, i] = 1
    print(f1_score(r1,r2,average='macro'))
    for i in range(28):
        print("%d,%d,%.5f"%(i, r3[:,i].sum().astype('int32'), f1_score(r1[:,i],r2[:,i])))
    
import numpy as np
import matplotlib.pyplot as plt
def plot_acc(accfile):
    with open(accfile) as f:
        q = f.readlines()
    train = []
    val = []
    for i in q:
        try:
            tmp = i.split(',')
            if tmp[0]=='val':
                val.append(float(tmp[3]))
            else:
                train.append(float(tmp[3]))
        except:
            pass
    if len(train)!=len(val):
        train = train[:len(val)]
    x = np.arange(len(val))
    val, train = np.array(val), np.array(train)
    plt.plot(x,train,label='train')
    plt.plot(x,val,label='val')
    plt.legend()
    plt.show()
    
        
        