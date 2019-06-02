#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt

def identity_equal(x):
    return x

def softmax(x):
    max_bia = np.max(x)
    exp_arr = np.exp(x - max_bia)
    sum_exp_arr = np.sum(exp_arr)
    y = exp_arr / sum_exp_arr
    return y



if __name__=='__main__':
    pass