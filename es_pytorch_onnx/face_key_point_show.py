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

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter 1 ~ 69 face num")
        raise SystemExit(1)

    plt.ion()
    landmark_face_dataset = pd.read_csv('~/Datasets/faces/faces/face_landmarks.csv')
    n = int(sys.argv[1])
    img_name = landmark_face_dataset.iloc[n, 0]
    img_landmark = np.asarray(landmark_face_dataset.iloc[n, 1:]).astype('float').reshape(-1, 2)
    img_dsc = io.imread('~/Datasets/faces/faces/' + img_name)
    plt.imshow(img_dsc)
    plt.scatter(img_landmark[:, 0], img_landmark[:, 1], s=10, marker='.', color='g')
    viz = Visdom(env='main')
    viz.matplot(plt)