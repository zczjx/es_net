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
    net = model_zoo.vision.mobilenet1_0(pretrained=True)
    net.hybridize()
    net.forward(mx.nd.ones((1,3,256,256)))
    net.export('demo_net')  



   
