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

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def label2image(pred):
    colormap = nd.array(VOC_COLORMAP, ctx=ctx, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)

    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    voc_dir = download_voc_pascal(data_dir='./data')
    crop_size = (320, 480)
    batch_size = 1000
    num_workers = 4
    num_classes = len(VOC_CLASSES)
    learning_rate = 0.1
    num_epochs = int(sys.argv[1])
    voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label)
    test_iter = gdata.DataLoader(voc_test, batch_size, last_batch='discard',
                                num_workers=num_workers)
    
    sym_filename = 'semanteme_rcnn-symbol.json'
    param_filename = 'semanteme_rcnn-0030.params'
    app_net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                param_filename, ctx=ctx)
    for batch_imgs, batch_labels in test_iter:
        pass
    # batch = voc_test[int(sys.argv[1]) : int(sys.argv[2])]
    imgs = batch_imgs[int(sys.argv[1]) : int(sys.argv[2])]
    labels = batch_labels[int(sys.argv[1]) : int(sys.argv[2])]

    for idx, image in enumerate(imgs):
        print('image.shape: ', image.shape)
        print('labels[idx].shape: ', labels[idx].shape)
        plt.subplot(len(imgs), 3, idx * 3 + 1)
        plt.imshow(image.transpose((1, 2, 0)).asnumpy())
        plt.subplot(len(imgs), 3, idx * 3 + 2)
        disp_label = label2image(labels[idx].as_in_context(ctx))
        plt.imshow(disp_label.asnumpy())
        plt.subplot(len(imgs), 3, idx * 3 + 3)
        in_val = image.expand_dims(axis=0)
        print('in_val.shape: ', in_val.shape)
        pred = nd.argmax(app_net(in_val.as_in_context(ctx)), axis=1)
        print('pred.shape: ', pred.shape)
        out_val = label2image(pred.reshape((pred.shape[1], pred.shape[2])))
        print('out_val.shape: ', out_val.shape)
        plt.imshow(out_val.asnumpy())
    plt.show()
 
