#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter onnx model file path")
        raise SystemExit(1)

    batch_size = 1000
    onnx_file  = sys.argv[1]
    input_shape = (batch_size, 1, 28, 28)
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    lenet = onnx_mxnet.import_to_gluon(onnx_file, ctx=ctx)

    start = time.time()
    test_acc = evaluate_accuracy(test_data_batched, lenet, ctx=ctx)
    print('test acc %.3f, ' 'time %.1f sec' % (test_acc, time.time() - start))

   
 
