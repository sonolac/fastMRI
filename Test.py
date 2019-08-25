#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:35:42 2019

@author: cong
"""
import numpy as np
import torch
import random
from models.unet import train_unet
args = train_unet.create_arg_parser().parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_unet.main(args)