#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from common_func import *

# ctx = try_gpu()
# ctx = mx.cpu()

def step_func(x):
    return nd.array(x > 0, dtype='int32', ctx=ctx)

def sigmoid(x):
    return 1 / (1 + nd.exp(-1 * x))

def ReLU(x):
    return nd.maximum(0, x)



if __name__=='__main__':
    xval = nd.arange(-5, 5, 0.1, ctx=ctx)
    yval = sigmoid(xval)
    zval = step_func(xval)
    uval = ReLU(xval)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.ylim(-1, 2)
    plt.title('active func')
    plt.plot(xval.asnumpy(), yval.asnumpy(), label="sigmoid")
    plt.plot(xval.asnumpy(), zval.asnumpy(), label="step_func")
    plt.plot(xval.asnumpy(), uval.asnumpy(), label="ReLU")
    plt.legend()
    plt.show()
    