#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *
from object_detect_ssd import *

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)
    batch_size, edge_size = 1000, 256
    ssd_net = obj_ssd(num_classes=1, import_sym=True, prefix='zcz_ssd')
    train_iter, val_iter = load_data_pikachu(batch_size, edge_size)
    batch = val_iter.next()
    imgs = batch.data[0][int(sys.argv[1]) : int(sys.argv[2])]
    labels = batch.label[0][int(sys.argv[1]) : int(sys.argv[2])]
    threshold = 0.5

    imgs_one_line = int(len(imgs) / 2 + (len(imgs) % 2))
    for idx, image in enumerate(imgs):
        in_val = image.expand_dims(axis=0)
        out_val = inference(X=in_val, net=ssd_net)
        plt.subplot(2, imgs_one_line, idx+1)
        disp_img = (image.transpose((1, 2, 0)).asnumpy()) / 255
        fig = plt.imshow(disp_img)
        for row in out_val:
            score = row[1].asscalar()
            if score < threshold:
                continue
            h, w = disp_img.shape[0:2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            show_bboxes(fig.axes, bbox, '%.2f' % score, 'r')
    plt.show()

