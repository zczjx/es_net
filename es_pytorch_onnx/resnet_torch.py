#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as func
from common_torch import *
import os, time, sys, pickle

class residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                        kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return func.relu(y + x)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(residual(out_channels, out_channels))
    return nn.Sequential(*blk)


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    batch_size = 100
    prefix = 'cnn_' + os.path.basename(__file__).split('.')[0]
    onnx_file  = prefix + '.onnx'
    input_shape = (batch_size, 1, 28, 28)
    dummy_input = torch.randn(batch_size, 1, 28, 28, requires_grad=True, device=device)
    input_names = ['demo_in_data']
    output_names = ['demo_out_data']

    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))
    print('filename: ', os.path.basename(__file__))

    app_net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                    stride=2, padding=3),
                nn.BatchNorm2d(num_features=64), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    app_net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    app_net.add_module("resnet_block2", resnet_block(64, 128, 2))
    app_net.add_module("resnet_block3", resnet_block(128, 256, 2))
    app_net.add_module("resnet_block4", resnet_block(256, 512, 2))
    app_net.add_module("global_avg_pool1", nn.AvgPool2d(kernel_size=1))
    app_net.add_module("dense10", conv_fc_out(in_channels=512, out_channels=10))

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