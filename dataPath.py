#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:34:50 2018

@author: qzane
"""

import platform
PCName = platform.node()
SAVE_PATH = './save'
import os
if not os.path.isdir(SAVE_PATH):
 os.makedirs(SAVE_PATH)


TRAIN_PATH = '/home/qzane/c/ml/pj/data/train'
VAL_PATH = '/home/qzane/c/ml/pj/data/val'



