#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata


if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    cifar10_test = gdata.vision.CIFAR10(train=False)
    print('type(cifar10_test)', type(cifar10_test))
    print('len(cifar10_test)', len(cifar10_test))
    imgs, labels = cifar10_test[int(sys.argv[1]) : int(sys.argv[2])]
    print('imgs.shape: ', imgs.shape)

    imgs_one_line = int(len(imgs) / 2 + (len(imgs) % 2))
    for idx in range(len(imgs)):
        plt.subplot(2, imgs_one_line, idx+1)
        plt.imshow(imgs[idx].asnumpy())
    plt.show()
