#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt

def mean_squared_err(mean_val, real_val):
    return 0.5 * np.sum((mean_val - real_val)**2)

def cross_entropy_err(mean_val, real_val):
    delta = 1e-7
    return -np.sum(real_val * np.log(mean_val + delta))


if __name__=='__main__':
    real = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    mean = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print ('real.ndim: ', real.ndim)
    print ('mean.ndim: ', mean.ndim)
    print('mean_squared_err: ', mean_squared_err(mean_val=mean, real_val=real))
    print('cross_entropy_err', cross_entropy_err(mean_val=mean, real_val=real))