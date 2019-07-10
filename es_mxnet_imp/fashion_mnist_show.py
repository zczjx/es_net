#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    box = (100, 100, 400, 400)
    region = pil_img.crop(box)
    region.show()
    pil_img.show()

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    mnist_test = gdata.vision.FashionMNIST(train=False)
    imgs, labels = mnist_test[int(sys.argv[1]) : int(sys.argv[2])]

    imgs_one_line = int(len(imgs) / 2 + (len(imgs) % 2))
    for idx in range(len(imgs)):
        pil_img = Image.fromarray(np.uint8(imgs[idx].reshape(28, 28).asnumpy()))
        plt.subplot(2, imgs_one_line, idx+1)
        plt.imshow(pil_img)
    plt.show()

    # print('old shape: ', img.shape)  # (784,)
    # img = img.reshape(28, 28).asnumpy()  # 把图像的形状变为原来的尺寸
    # print('new shape: ', img.shape)  # (28, 28)
    # img_show(img)