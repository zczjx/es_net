#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides,
                padding, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk

if __name__=='__main__':
    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    nin_net = nn.Sequential()
    nin_net.add(nin_block(24, kernel_size=5, strides=2, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(64, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(96, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=2, strides=1), nn.Dropout(0.5),
                nin_block(10, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())

    '''
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    nin_net.initialize()
    for blk in nin_net:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
    '''

    lr = 0.05
    num_epochs = 100
    nin_net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(nin_net.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=nin_net, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)
    