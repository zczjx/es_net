#!/usr/bin/env python3
# coding: utf-8
import sys, os

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.onnx
from onnx import checker
import onnx
import logging

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    prefix = 'torch_net'
    onnx_file  = prefix + '.onnx'
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    input_names = ['demo_in_data']
    output_names = ['demo_out_data']

    model_net = models.resnet18(pretrained=True).cuda()
    model_net.eval()
    torch.onnx.export(model_net, dummy_input, onnx_file, verbose=False,
        input_names=input_names, output_names=output_names)

    # Load onnx model
    model_proto = onnx.load(onnx_file)

    # Check that the IR is well formed
    checker.check_model(model_proto)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model_proto.graph)

