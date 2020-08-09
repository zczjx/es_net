#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torchvision import transforms
import os, time, sys, pickle, getopt, argparse
sys.path.append(os.path.abspath('..'))
from es_pytorch_onnx.common_torch import *
from voc_dataset import *

parser = argparse.ArgumentParser(
    description='show the VOC dataset')

parser.add_argument('-l', '--label', action='store_true',
                    help='show label in output image')

parser.add_argument('-b', '--bbox', action='store_true',
                    help='show bbox in output image')

parser.add_argument('num', nargs='?', default=4, type=int,
                    help='num of images to show')

args = parser.parse_args()

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the images num or -l -b to enable label bbox, exam: 4 -l -b")
        raise SystemExit(1)

    trans_func = transforms.ToPILImage()
    batch_size = 100
    # voc2012_train_iter = load_data_vocdetection(batch_size=batch_size, image_set='train', year='2012')
    width = 320
    height = 240
    # train_iter, validate_iter = load_data_pikachu(batch_size, edge_size)
    voc2012_val_iter = load_vocdetection_format_dataset(batch_size=batch_size,
                                                        height=height, width=width,
                                                        image_set='val', year='2012')
    # print('len(voc2012_train_iter): ', len(voc2012_train_iter))
    # print('type(voc2012_train_iter): ', type(voc2012_train_iter))
    print('len(voc2012_val_iter): ', len(voc2012_val_iter))
    print('type(voc2012_val_iter): ', type(voc2012_val_iter))
    imgs_batch, labels_batch = iter(voc2012_val_iter).next()
    print('len(imgs_batch): ', len(imgs_batch))
    print('type(imgs_batch): ', type(imgs_batch))
    print('imgs_batch.shape: ', imgs_batch.shape)
    print('len(labels_batch): ', len(labels_batch))
    print('type(labels_batch): ', type(labels_batch))
    imgs_one_line = int(args.num / 2 + (args.num % 2))
    for idx in range(0, args.num):
        data = imgs_batch[idx]
        labels = labels_batch[idx]
        # print('len(data): ', len(data))
        # print('type(data): ', type(data))
        # print('data.size(): ', data.size())
        # print('type(labels): ', type(labels))
        # print('len(labels): ', len(labels))
        img = data.squeeze(0)
        img_plt = trans_func(img).convert('RGB')
        axes = plt.subplot(2, imgs_one_line, (idx + 1))
        i = 0
        for item in labels:
            # print('item[0]: ', int(item[0].item()))
            name = voc_classes[int(item[0].item())]
            xmin = int(item[1].item() * width)
            ymin = int(item[2].item() * height)
            xmax = int(item[3].item() * width)
            ymax = int(item[4].item() * height)
            # print('name: ', name)
            # print('(xmin, ymin, xmax, ymax): ', xmin, ymin, xmax, ymax)
            i += 1
            i %= len(color_list)
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                    linewidth=2, edgecolor=color_list[i], fill=False)
            if args.bbox == True:
                axes.add_patch(rect)

            if args.label == True:
                axes.text(rect.xy[0], rect.xy[1], name,
                        va='center', ha='center', color='k',
                        bbox=dict(facecolor='w'))
        plt.imshow(img_plt)
        plt.axis('off')
        plt.ioff()

    visom_show(plt=plt)