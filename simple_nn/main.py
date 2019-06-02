#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
from active_func import *
from es_net import *

if __name__=='__main__':
    dnn = es_net(layers=3, active_func=sigmoid)
    dnn.setup_nn_param(0, 
            weight_arr=np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
            bias_arr=np.array([0.1, 0.2, 0.3]))
    dnn.setup_nn_param(1, 
            weight_arr=np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
            bias_arr=np.array([0.1, 0.2]))
    dnn.setup_nn_param(2, 
            weight_arr=np.array([[0.1, 0.3], [0.2, 0.4]]),
            bias_arr=np.array([0.1, 0.2]))
    
    x_arr = np.array([0.1, 0.5])
    y_arr = dnn.forward_inference(input_x=x_arr)
    print(y_arr)


