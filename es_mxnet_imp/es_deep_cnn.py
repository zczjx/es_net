#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys, os
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *
import time


if __name__=='__main__':
    batch_size = 100
    train_data_batched, test_data_batched = load_data_mnist(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))

    es_deep_cnn = nn.Sequential()
    es_deep_cnn.add(nn.Conv2D(channels=30, kernel_size=5, activation='relu'),
            nn.Conv2D(channels=30, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(100, activation='relu'),
            nn.Dense(10))
    lr = 0.1
    num_epochs = 10
    es_deep_cnn.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(es_deep_cnn.collect_params(), 'sgd', {'learning_rate': lr})
    do_train(net=es_deep_cnn, 
        train_iter=train_data_batched, test_iter=test_data_batched, 
        batch_size=batch_size, trainer=trainer, 
        num_epochs=num_epochs, ctx=ctx)


    






    
