#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn, model_zoo
from mxnet.gluon import data as gdata
from common_mx import *

if __name__=='__main__':
    prefix = 'demo_net'
    syms = prefix + '-symbol.json'
    params = prefix + '-0000.params'
    onnx_file  = prefix + '.onnx'
    input_shape = (1,3,28,28)

    lenet = nn.HybridSequential()
    lenet.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))

    lenet.initialize(init=init.Xavier(), ctx=ctx)
    lenet.hybridize()
    lenet(mx.nd.ones(input_shape, ctx=ctx))
    lenet.export(prefix)

    # convert to onnx
    onnx_model_path = onnx_mxnet.export_model(syms, params, [input_shape], np.float32, onnx_file)

    # Load onnx model
    model_proto = onnx.load_model(onnx_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)



   
