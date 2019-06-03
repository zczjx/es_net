#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the image path in argv")
        raise SystemExit(1)

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_test[int(sys.argv[1])]
    label = t_test[int(sys.argv[1])]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
    print(img.shape)  # (28, 28)

    img_show(img)
