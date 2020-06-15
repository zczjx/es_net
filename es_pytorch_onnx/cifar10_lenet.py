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

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(in_channels=16, out_channels=400, kernel_size=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(in_channels=400, out_channels=120, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=120, out_channels=84, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=10, kernel_size=1))

    def forward(self, img):
        feature = self.conv(img)
        # print('feature.shape: ', feature.shape)
        # print('img.shape: ', img.shape)
        # print('feature.view(img.shape[0], -1).shape: ', feature.view(img.shape[0], -1).shape)
        # output = self.fc(feature)
        # feature = feature.view(feature.shape[0], -1)
        return torch.squeeze(feature)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

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

    app_net = lenet()

    '''
    app_net.eval()
    X = torch.rand(batch_size, 1, 28, 28)
    X = app_net(X)
    print('output shape: ', X.shape)
    exit(1)
    '''


    lr = 0.001
    num_epochs = int(sys.argv[1])
    optimizer  = torch.optim.Adam(app_net.parameters(), lr=lr)
    test_acc_list = do_train(net=app_net,
                            train_iter=train_data_batched, test_iter=test_data_batched,
                            batch_size=batch_size, optimizer=optimizer,
                            num_epochs=num_epochs, device=device)
    app_net.eval()

    torch.onnx.export(app_net, dummy_input, onnx_file,
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
