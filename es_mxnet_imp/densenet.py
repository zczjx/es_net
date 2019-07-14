#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

def cov_norm_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    
    return blk

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=1))
    
    return blk

class denseblock(nn.Block):
    
    def __init__(self, num_convs, num_channels, **kwargs):
        super(denseblock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(cov_norm_block(num_channels))
       
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = nd.concat(x, y, dim=1)
        return x
       

  
if __name__=='__main__':
    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    densenet = nn.Sequential()
    densenet.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
       densenet.add(denseblock(num_convs, growth_rate))
       num_channels += num_convs * growth_rate
       if i != len(num_convs_in_dense_blocks) - 1:
           num_channels //= 2
           densenet.add(transition_block(num_channels))

    densenet.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(),
                nn.Dense(10))
    '''
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    densenet.initialize()
    for blk in densenet:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
    exit()
    '''

    lr = 0.05
    num_epochs = 30
    densenet.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(densenet.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=densenet, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)
    