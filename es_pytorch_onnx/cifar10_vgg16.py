#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from common_torch import *
import os, time, sys, pickle

class_name = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    update_in_channels = in_channels
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels=update_in_channels,
                              out_channels=out_channels,
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True)]
        update_in_channels = out_channels

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, vgg_arch_tuple):
        super(VGG, self).__init__()
        self.features = self._make_vgg_blocks(vgg_arch_tuple)
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 10))

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        update_in_channels = in_channels
        for _ in range(num_convs):
            layers += [nn.Conv2d(in_channels=update_in_channels,
                                out_channels=out_channels,
                                kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_features=out_channels),
                        nn.ReLU(inplace=True)]
            update_in_channels = out_channels

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _make_vgg_blocks(self, vgg_arch_tuple):
        layers = []

        for num_convs, in_channels, out_channels in vgg_arch_tuple:
            layers += [self.vgg_block(num_convs, in_channels, out_channels)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    tboard_writer = SummaryWriter('runs/cifar10_vgg16')
    batch_size = 100
    prefix = 'cnn_' + os.path.basename(__file__).split('.')[0]
    onnx_file  = prefix + '.onnx'
    input_shape = (batch_size, 3, 32, 32)
    dummy_input = torch.randn(batch_size, 3, 32, 32, requires_grad=True, device=device)
    input_names = ['demo_in_data']
    output_names = ['demo_out_data']

    train_data_batched, test_data_batched = load_data_cifar10(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))
    print('filename: ', os.path.basename(__file__))
    vgg_16_arch_tuple = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
    vgg16 = VGG(vgg_16_arch_tuple)

    '''
    vgg16.eval()
    X = torch.rand(batch_size, 3, 32, 32)
    X = vgg16(X)
    print('output shape: ', X.shape)
    exit(1)
    '''


    lr = 0.001
    num_epochs = int(sys.argv[1])
    optimizer  = torch.optim.Adam(vgg16.parameters(), lr=lr)
    test_acc_list = do_train(net=vgg16,
                            train_iter=train_data_batched, test_iter=test_data_batched,
                            batch_size=batch_size, optimizer=optimizer,
                            num_epochs=num_epochs, device=device, tboard_writer=tboard_writer)
    vgg16.eval()
    images, labels = iter(test_data_batched).next()
    tboard_writer.add_graph(vgg16, images.to(device=device))
    tboard_writer.close()

    torch.onnx.export(vgg16, dummy_input, onnx_file,
        input_names=input_names, output_names=output_names,
        verbose=False)

    # Load onnx model
    model_proto = onnx.load(onnx_file)

    # Check that the IR is well formed
    checker.check_model(model_proto)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model_proto.graph)
