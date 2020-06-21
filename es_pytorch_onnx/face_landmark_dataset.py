#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, sys, pickle
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from visdom import Visdom

class face_landmark_dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmark_face_dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(landmark_face_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.landmark_face_dataset.iloc[n, 0]
        img_path = self.root_dir + img_name
        img = io.imread(img_path)
        img_landmark = np.asarray(self.landmark_face_dataset.iloc[n, 1:]).astype('float').reshape(-1, 2)
        sample = {'image': img, 'landmark': img_landmark}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter 1 ~ 69 face num")
        raise SystemExit(1)

    plt.ion()
    face_dataset = face_landmark_dataset(csv_file='/home/clarencez/Datasets/faces/faces/face_landmarks.csv',
                                         root_dir='/home/clarencez/Datasets/faces/faces/')

    n = int(sys.argv[1])
    item = face_dataset[n]
    plt.axis('off')
    plt.ioff()
    plt.imshow(item['image'])
    landmark = item['landmark']
    plt.scatter(landmark[:, 0], landmark[:, 1], s=10, marker='.', color='b')
    viz = Visdom(env='main')
    viz.matplot(plt)