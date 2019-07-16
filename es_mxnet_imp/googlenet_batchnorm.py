#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

class inception(nn.Block):

    def __init__(self, ch1, ch2, ch3, ch4, **kwargs):
        super(inception, self).__init__(**kwargs)
        self.path1_1 = nn.Conv2D(channels=ch1, kernel_size=1, activation='relu')

        self.path2_1 = nn.Conv2D(channels=ch2[0], kernel_size=1, activation='relu')
        self.path2_2 = nn.Conv2D(channels=ch2[1], kernel_size=3, padding=1, 
                                activation='relu')
        
        self.path3_1 = nn.Conv2D(channels=ch3[0], kernel_size=1, activation='relu')
        self.path3_2 = nn.Conv2D(channels=ch3[1], kernel_size=5, padding=2, 
                                activation='relu')

        self.path4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.path4_2 = nn.Conv2D(channels=ch4, kernel_size=1, activation='relu')
    
    def forward(self, x):
        p1 = self.path1_1(x)
        p2 = self.path2_2(self.path2_1(x))
        p3 = self.path3_2(self.path3_1(x))
        p4 = self.path4_2(self.path4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    blk1 = nn.Sequential()
    blk1.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    
    blk2 = nn.Sequential()
    blk2.add(nn.Conv2D(channels=64, kernel_size=1),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.Conv2D(channels=192, kernel_size=3, padding=1),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    blk3 = nn.Sequential()
    blk3.add(inception(64, (96, 128), (16, 32), 32),
            inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    
    blk4 = nn.Sequential()
    blk4.add(inception(192, (96, 208), (16, 48), 64),
            inception(160, (112, 224), (24, 64), 64),
            inception(128, (128, 256), (24, 64), 64),
            inception(112, (144, 288), (32, 64), 64),
            inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    
    blk5 = nn.Sequential()
    blk5.add(inception(256, (160, 320), (32, 128), 128),
            inception(384, (192, 384), (48, 128), 128),
            nn.GlobalAvgPool2D())
    googlenet = nn.Sequential()
    googlenet.add(blk1, blk2, blk3, blk4, blk5, nn.Dense(10))


    '''
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    googlenet.initialize()
    for blk in googlenet:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
    
    '''
    
    lr = 0.05
    num_epochs = int(sys.argv[1])
    googlenet.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(googlenet.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=googlenet, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)
    