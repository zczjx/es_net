#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *
import os, time, sys, pickle


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))
    print('filename: ', os.path.basename(__file__))

    lenet = nn.Sequential()
    lenet.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))
    lr = 0.05
    num_epochs = int(sys.argv[1])
    lenet.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(lenet.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=lenet, 
                            train_iter=train_data_batched, test_iter=test_data_batched, 
                            batch_size=batch_size, trainer=trainer, 
                            num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)


    
