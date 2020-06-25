#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from PIL import Image
import os, time, sys, pickle, tarfile, zipfile
import logging
from onnx import checker
import onnx
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
import seaborn
seaborn.set()

def try_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

device = try_gpu()
# logging.basicConfig(level=logging.INFO)

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def load_data_cifar10(batch_size, resize=None, root='~/Datasets/CIFAR10'):
    """Download the cifar-10 dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    cifar10_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def load_data_cifar100(batch_size, resize=None, root='~/Datasets/CIFAR100'):
    """Download the cifar-100 dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    cifar100_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n

def do_train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    test_acc_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            '''
            print('len(y): ', len(y))
            print('len(y_hat): ', len(y_hat))
            print('y.size(): ', y.size())
            print('y_hat.size(): ', y_hat.size())
            print('type(y): ', type(y))
            print('type(y_hat): ', type(y_hat))
            '''
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device=device)
        test_acc_list.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    return test_acc_list

class conv_fc_out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv_fc_out, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(
                        in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1))
    def forward(self, x):
        feature = self.conv(x)
        return torch.squeeze(feature)

def visom_show(plt=None):
    viz = Visdom(env='main')
    viz.matplot(plt)