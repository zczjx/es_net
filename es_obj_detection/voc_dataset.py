#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, sys, pickle, getopt
sys.path.append(os.path.abspath('..'))
import random
import torch
from torch import nn, optim
from torchvision import transforms
from es_pytorch_onnx.common_torch import *
import threading

voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

class dataset_process_pool(object):
    def __init__(self, num_thread, dataset, height, width):
        self.num_thread = num_thread
        self.dateset = dataset
        self.thread_pool = [None] * self.num_thread
        self.data_list_pool = []
        self.sub_len = int(len(dataset) / self.num_thread)
        self.height = height
        self.width = width

        start = 0
        for i in range(0, num_thread-1):
            self.data_list_pool.append(list())
            self.thread_pool[i] = threading.Thread(
                    target=self.dateset_process_task,
                    kwargs={'id_thread':i, 'start_idx':start,
                            'end_idx':(start+self.sub_len),
                            'data_sublist': self.data_list_pool[i]})
            start += self.sub_len + 1

        self.data_list_pool.append(list())
        self.thread_pool[num_thread-1] = threading.Thread(
                    target=self.dateset_process_task,
                    kwargs={'id_thread':(num_thread-1), 'start_idx':start,
                            'end_idx':(len(dataset) - 1),
                            'data_sublist': self.data_list_pool[num_thread-1]})


    def dateset_process_task(self, id_thread, start_idx, end_idx, data_sublist=[]):
        local_id = id_thread
        for idx in range(start_idx, end_idx+1):
            # print('idx: ', idx)
            img, annotation = self.dateset[idx]
            # print(annotation)
            width_orig = float(annotation['annotation']['size']['width'])
            height_orig = float(annotation['annotation']['size']['height'])
            width_ratio = self.width / width_orig
            height_ratio = self.height / height_orig
            labels_list = []
            for item in annotation['annotation']['object']:
                class_idx = voc_classes.index(item['name'])
                xmin = int(int(item['bndbox']['xmin']) * width_ratio)
                ymin = int(int(item['bndbox']['ymin']) * height_ratio)
                xmax = int(int(item['bndbox']['xmax']) * width_ratio)
                ymax = int(int(item['bndbox']['ymax']) * height_ratio)
                obj_label = [class_idx, xmin, ymin, xmax, ymax]
                labels_list.append(obj_label)
            data_sublist.append(tuple((img, torch.tensor(labels_list))))

    def get_dataset_list(self):
        for idx in range(0, self.num_thread):
            self.thread_pool[idx].start()
        for idx in range(0, self.num_thread):
            self.thread_pool[idx].join()

        return self.data_list_pool

def load_vocdetection_format_dataset(height=256, width=256, root='~/Datasets/VOCDetection', image_set='train', year='2012'):
    """Download the VOC Detection dataset and then load into memory."""
    trans = []

    trans.append(torchvision.transforms.Resize(size=(height, width)))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    voc_train = torchvision.datasets.VOCDetection(root, year=year,
                                                image_set=image_set,
                                                download=False,
                                                transform=transform)
    dataset_work = dataset_process_pool(num_thread=8, dataset=voc_train,
                                        height=height, width=width)
    tmp_list = dataset_work.get_dataset_list()
    dataset_list = []
    for list_item in tmp_list:
        dataset_list += list_item

    for idx in range(0, len(dataset_list)):
        # do shuffle
        idx_random = random.randint(0, len(dataset_list)-1)
        dataset_list[idx], dataset_list[idx_random] = dataset_list[idx_random], dataset_list[idx]

    return dataset_list