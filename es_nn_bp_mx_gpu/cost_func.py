#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
from common_func import *

def mean_squared_err(output_val, label):
    return 0.5 * nd.sum((output_val - label)**2)

def cross_entropy_err(output_val, label):
    if output_val.ndim == 1:
        label = label.reshape(1, label.size)
        output_val = output_val.reshape(1, output_val.size)
    
    delta = 1e-7
    batch_size = output_val.shape[0]
    return -nd.sum(nd.log(output_val[nd.arange(batch_size, ctx=ctx), label] + delta)) / batch_size


if __name__=='__main__':
    label = nd.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ctx=ctx)
    output_val = nd.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], ctx=ctx)
    print ('output_val.ndim: ', output_val.ndim)
    print ('label.ndim: ', label.ndim)
    print('mean_squared_err: ', mean_squared_err(output_val=output_val, label=label))
    print('cross_entropy_err', cross_entropy_err(output_val=output_val, label=label))