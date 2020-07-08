#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from common_torch import *
from torchvision import transforms
import os, time, sys, pickle, getopt

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the images num or -l -b to enable label bbox, exam: 4 -l -b")
        raise SystemExit(1)

    num = int(sys.argv[1])
    opts, args = getopt.getopt(sys.argv[2:], "bl")
    enable_label = False
    enable_bbox = False
    for opt, arg in opts:
        if opt == '-l':
            enable_label = True
        if opt == '-b':
            enable_bbox = True

    trans_func = transforms.ToPILImage()
    batch_size = 1
    # voc2012_train_iter = load_data_vocdetection(batch_size=batch_size, image_set='train', year='2012')
    voc2012_val_iter = load_data_vocdetection(image_set='val', year='2012')
    # print('len(voc2012_train_iter): ', len(voc2012_train_iter))
    # print('type(voc2012_train_iter): ', type(voc2012_train_iter))
    print('len(voc2012_val_iter): ', len(voc2012_val_iter))
    print('type(voc2012_val_iter): ', type(voc2012_val_iter))
    imgs_one_line = int(num / 2 + (num % 2))
    for idx in range(0, num):
        data, target = iter(voc2012_val_iter).next()
        print('len(data): ', len(data))
        print('type(data): ', type(data))
        print('data.size(): ', data.size())
        # print('target: ', target)
        img = data.squeeze(0)
        img_plt = trans_func(img).convert('RGB')
        axes = plt.subplot(2, imgs_one_line, (idx + 1))
        idx = 0
        for item in target['annotation']['object']:
            name = item['name'][0]
            xmin = int(item['bndbox']['xmin'][0])
            ymin = int(item['bndbox']['ymin'][0])
            xmax = int(item['bndbox']['xmax'][0])
            ymax = int(item['bndbox']['ymax'][0])
            print('name: ', name)
            print('(xmin, ymin, xmax, ymax): ', xmin, ymin, xmax, ymax)
            idx += 1
            idx %= len(color_list)
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                    linewidth=2, edgecolor=color_list[idx], fill=False)
            if enable_bbox == True:
                axes.add_patch(rect)

            if enable_label == True:
                axes.text(rect.xy[0], rect.xy[1], name,
                        va='center', ha='center', color='k',
                        bbox=dict(facecolor='w'))
        plt.imshow(img_plt)
        plt.axis('off')
        plt.ioff()

    visom_show(plt=plt)