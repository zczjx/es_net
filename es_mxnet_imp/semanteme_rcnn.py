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

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the epochs exam: 20")
        raise SystemExit(1)

    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    voc_dir = download_voc_pascal(data_dir='./data')
    crop_size = (320, 480)
    batch_size = 10
    num_workers = 4
    num_classes = len(VOC_CLASSES)
    learning_rate = 0.1
    num_epochs = int(sys.argv[1])
    voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label)
    voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label)
    train_iter = gdata.DataLoader(voc_train, batch_size, shuffle=True,
                                last_batch='discard', num_workers=num_workers)
    test_iter = gdata.DataLoader(voc_test, batch_size, last_batch='discard',
                                num_workers=num_workers)
    pretrained_resnet =  model_zoo.vision.resnet18_v2(pretrained=True)
    app_net = nn.HybridSequential()

    for layer in pretrained_resnet.features[:-2]:
        app_net.add(layer)

    app_net.add(nn.Conv2D(channels=num_classes, kernel_size=1),
                nn.Conv2DTranspose(channels=num_classes, kernel_size=64, 
                        padding=16, strides=32))
    
    # for blk in app_net:
    #    print(blk.name)
    
    app_net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)))
    app_net[-2].initialize(init=init.Xavier())
    app_net.collect_params().reset_ctx(ctx=ctx)
    app_net.hybridize()
    trainer = gluon.Trainer(app_net.collect_params(), 'sgd',
                {'learning_rate': learning_rate, 'wd': 1e-3})
    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
    test_acc_list = common_train(net=app_net, 
                        train_iter=train_iter, test_iter=test_iter, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, loss_func=loss, ctx=ctx)

    app_net.export('semanteme_rcnn', epoch=num_epochs)



    # print('len(train_iter): ', len(train_iter))
    # for X, Y in train_iter:
    #    print(X.shape)
    #    print(Y.shape)
    