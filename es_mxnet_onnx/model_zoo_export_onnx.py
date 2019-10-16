#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import data as gdata
from common_mx import *

if __name__=='__main__':
    prefix = 'model_zoo'
    syms = prefix + '-symbol.json'
    params = prefix + '-0000.params'
    onnx_file  = prefix + '.onnx'
    input_shape = (1, 3, 224, 224)

    mobile_net = vision.mobilenet_v2_1_0(pretrained=True, ctx=ctx)
    mobile_net.hybridize()
    mobile_net(mx.nd.ones(input_shape, ctx=ctx))
    mobile_net.export(prefix)

    # convert to onnx
    onnx_model_path = onnx_mxnet.export_model(syms, params, [input_shape], np.float32, onnx_file)

    # Load onnx model
    model_proto = onnx.load_model(onnx_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)



   
