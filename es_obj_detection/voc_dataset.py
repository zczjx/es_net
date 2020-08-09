#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, sys, pickle, getopt
sys.path.append(os.path.abspath('..'))
import random
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import BatchSampler, Dataset, DataLoader
from es_pytorch_onnx.common_torch import *
import threading

voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def voc_detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    # print('len(batch): ', len(batch))
    # print('type(batch): ', type(batch))
    # print('batch.size(): ', batch.size())
    targets = []
    imgs = []
    for sample in batch:
        # print('len(sample): ', len(sample))
        # print('type(sample): ', type(sample))
        # print('sample.size(): ', sample.size())
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


def load_vocdetection_format_dataset(batch_size=10, height=256, width=256,
                                     root='~/Datasets/VOCDetection',
                                     image_set='train', year='2012'):
    """Download the VOC Detection dataset and then load into memory."""
    trans = []

    trans.append(torchvision.transforms.Resize(size=(height, width)))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    voc_train = torchvision.datasets.VOCDetection(root, year=year,
                                                image_set=image_set,
                                                download=False,
                                                transform=transform)
    obj_detect_dataset = VocObjDetectDataset(dataset=voc_train,
                                             height=height, width=width)
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
    train_iter = torch.utils.data.DataLoader(obj_detect_dataset, batch_size=batch_size,
                                             shuffle=True, pin_memory=True,
                                             num_workers=num_workers, collate_fn=voc_detection_collate)
    return train_iter

class VocObjDetectDataset(Dataset):
    """VocObjDetect dataset."""

    def __init__(self, dataset, height, width):
        self.num_thread = 8
        self.dateset = dataset
        self.thread_pool = [None] * self.num_thread
        self.data_list_pool = []
        self.sub_len = int(len(dataset) / self.num_thread)
        self.height = height
        self.width = width

        start = 0
        for i in range(0, self.num_thread-1):
            self.data_list_pool.append(list())
            self.thread_pool[i] = threading.Thread(
                    target=self.dateset_process_task,
                    kwargs={'id_thread':i, 'start_idx':start,
                            'end_idx':(start+self.sub_len),
                            'data_sublist': self.data_list_pool[i]})
            start += self.sub_len + 1

        self.data_list_pool.append(list())
        self.thread_pool[self.num_thread-1] = threading.Thread(
                    target=self.dateset_process_task,
                    kwargs={'id_thread':(self.num_thread-1), 'start_idx':start,
                            'end_idx':(len(dataset) - 1),
                            'data_sublist': self.data_list_pool[self.num_thread-1]})

        tmp_list = self.get_dataset_list()
        self.processed_dataset= []
        for list_item in tmp_list:
            self.processed_dataset += list_item
        print('len(self.processed_dataset: ', len(self.processed_dataset))
        print('type(self.processed_dataset): ', type(self.processed_dataset))

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, labels = self.processed_dataset[idx]

        return img, labels

    def dateset_process_task(self, id_thread, start_idx, end_idx, data_sublist=[]):
        local_id = id_thread
        for idx in range(start_idx, end_idx+1):
            # print('idx: ', idx)
            img, annotation = self.dateset[idx]
            # print(annotation)
            width_orig = float(annotation['annotation']['size']['width'])
            height_orig = float(annotation['annotation']['size']['height'])
            labels_list = []
            for item in annotation['annotation']['object']:
                class_idx = voc_classes.index(item['name'])
                xmin = float(int(item['bndbox']['xmin']) / width_orig)
                ymin = float(int(item['bndbox']['ymin']) / height_orig)
                xmax = float(int(item['bndbox']['xmax']) / width_orig)
                ymax = float(int(item['bndbox']['ymax']) / height_orig)
                obj_label = [class_idx, xmin, ymin, xmax, ymax]
                labels_list.append(obj_label)
            # labels_list = torch.tensor(labels_list)
            # labels_list = torch.unsqueeze(labels_list, 0)
            data_sublist.append(tuple((img, torch.tensor(labels_list, dtype=torch.float))))

    def get_dataset_list(self):
        for idx in range(0, self.num_thread):
            self.thread_pool[idx].start()
        for idx in range(0, self.num_thread):
            self.thread_pool[idx].join()

        return self.data_list_pool