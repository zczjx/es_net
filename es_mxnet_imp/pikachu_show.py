#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    batch_size, edge_size = 1000, 256
    train_iter, val_iter = load_data_pikachu(batch_size, edge_size)
    batch = val_iter.next()
    print('len(batch.data): ', len(batch.data))
    print('len(batch.label): ', len(batch.label))
    print('batch.data[0].shape: ', batch.data[0].shape)
    print('batch.label[0].shape: ', batch.label[0].shape)
    imgs = (batch.data[0][int(sys.argv[1]) : int(sys.argv[2])].transpose((0, 2, 3, 1))) / 255
    labels = batch.label[0][int(sys.argv[1]) : int(sys.argv[2])]
    print('imgs.shape: ', imgs.shape)
    print('labels.shape: ', labels.shape)
    print('labels: ', labels)

    imgs_one_line = int(len(imgs) / 2 + (len(imgs) % 2))
    for idx, image in enumerate(imgs):
        plt.subplot(2, imgs_one_line, idx+1)
        fig = plt.imshow(image.asnumpy())
        box = labels[idx]
        show_bboxes(fig.axes, [box[0][1:5] * edge_size], colors=['lime'])
    plt.show()
