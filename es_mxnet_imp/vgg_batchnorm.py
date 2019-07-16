#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(), nn.Activation('relu'))
    blk.add(nn.MaxPool2D(pool_size=1, strides=2))
    return blk

def vgg(vgg_arch_tuple):
    net = nn.Sequential()

    for num_convs, num_channels in vgg_arch_tuple:
        net.add(vgg_block(num_convs, num_channels))
    
    net.add(nn.Dense(4096), nn.Dropout(0.5),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.Dense(4096), nn.Dropout(0.5),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.Dense(10))
    return net



if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    vgg_11_arch_tuple = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
    vgg_11 = vgg(vgg_11_arch_tuple)

    '''
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    vgg_11.initialize()
    for blk in vgg_11:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
    '''
    lr = 0.05
    num_epochs = int(sys.argv[1])
    vgg_11.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(vgg_11.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=vgg_11, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)
    

    
