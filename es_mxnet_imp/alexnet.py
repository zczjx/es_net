#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

if __name__=='__main__':
    batch_size=100
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))

    alexnet = nn.Sequential()
    alexnet.add(nn.Conv2D(channels=96, kernel_size=5, strides=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=1),
            nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=1),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    '''
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    alexnet.initialize()
    for layer in alexnet:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)
    '''
    lr = 0.05
    num_epochs = 100
    alexnet.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(alexnet.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=alexnet, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)

    
