#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import mxnet as mx
from mxnet import contrib, autograd, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the images path and x y pos")
        raise SystemExit(1)

    np.set_printoptions(2)
    img = image.imread(sys.argv[1]).asnumpy()
    print('img.shape: ', img.shape)
    fig = plt.imshow(img)
    h, w = img.shape[0:2]
    display_anchors(fig, w, h, 1, 1, s=[0.6])
    plt.show()
