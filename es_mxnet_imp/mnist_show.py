#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata


if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    mnist_test = gdata.vision.MNIST(train=False)
    print('type(mnist_test)', type(mnist_test))
    print('len(mnist_test)', len(mnist_test))
    imgs, labels = mnist_test[int(sys.argv[1]) : int(sys.argv[2])]

    imgs_one_line = int(len(imgs) / 2 + (len(imgs) % 2))
    for idx in range(len(imgs)):
        pil_img = Image.fromarray(np.uint8(imgs[idx].reshape(28, 28).asnumpy()))
        plt.subplot(2, imgs_one_line, idx+1)
        plt.imshow(pil_img)
    plt.show()
