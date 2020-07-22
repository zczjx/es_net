#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, sys, pickle, getopt
sys.path.append(os.path.abspath('..'))
import random
import torch
from torch import nn, optim
from torchvision import transforms
from es_pytorch_onnx.common_torch import *

voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def load_vocdetection_format_dataset(width=256, height=256, root='~/Datasets/VOCDetection', image_set='train', year='2012'):
    """Download the VOC Detection dataset and then load into memory."""
    trans = []

    trans.append(torchvision.transforms.Resize(size=(width, height)))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    voc_train = torchvision.datasets.VOCDetection(root, year=year,
                                                image_set=image_set,
                                                download=False,
                                                transform=transform)
    dataset_list = []
    for idx in range(0, len(voc_train)):
        img, annotation = voc_train[idx]
        # print(annotation)
        width_orig = float(annotation['annotation']['size']['width'])
        height_orig = float(annotation['annotation']['size']['height'])
        width_ratio = width / width_orig
        height_ratio = height / height_orig
        # print('width_orig: ', width_orig, 'height_orig: ', height_orig)
        # print('width_ratio: ', width_ratio, 'height_ratio: ', height_ratio)
        labels_list = []
        for item in annotation['annotation']['object']:
            class_idx = voc_classes.index(item['name'])
            # print('obj name: ', name, ' type(name): ', type(name))
            xmin = int(int(item['bndbox']['xmin']) * width_ratio)
            ymin = int(int(item['bndbox']['ymin']) * height_ratio)
            xmax = int(int(item['bndbox']['xmax']) * width_ratio)
            ymax = int(int(item['bndbox']['ymax']) * height_ratio)
            obj_label = [class_idx, xmin, ymin, xmax, ymax]
            labels_list.append(obj_label)
        dataset_list.append(tuple((img, labels_list)))

    for idx in range(0, len(dataset_list)):
        # do shuffle
        idx_random = random.randint(0, len(dataset_list)-1)
        dataset_list[idx], dataset_list[idx_random] = dataset_list[idx_random], dataset_list[idx]

    return dataset_list