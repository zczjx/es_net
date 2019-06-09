#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt

def step_func(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def ReLU(x):
    return np.maximum(0, x)



if __name__=='__main__':
    xval = np.arange(-5, 5, 0.1)
    yval = sigmoid(xval)
    zval = step_func(xval)
    uval = ReLU(xval)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.ylim(-1, 2)
    plt.title('active func')
    plt.plot(xval, yval, label="sigmoid")
    plt.plot(xval, zval, label="step_func")
    plt.plot(xval, uval, label="ReLU")
    plt.legend()
    plt.show()
    