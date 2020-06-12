#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from common_torch import *
import os, time, sys, pickle

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    batch_size = 10000
    cifar10_test = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10', train=False, download=True)
    print('len(cifar10_test): ', len(cifar10_test))
    print('type(cifar10_test): ', type(cifar10_test))
    head_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    arr_len = end_idx - head_idx + 1

    imgs_one_line = int(arr_len / 2 + (arr_len % 2))
    for idx in range(head_idx, end_idx + 1):
        plt.subplot(2, imgs_one_line, (idx - head_idx + 1))
        img, label = cifar10_test[idx]
        plt.imshow(img)
    visom_show(plt=plt)