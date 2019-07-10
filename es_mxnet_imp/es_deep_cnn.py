#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import d2lzh as d2l
import sys, os
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
import time

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()

def do_train(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))

def load_data_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.MNIST(root=root, train=True)
    mnist_test = gdata.vision.MNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter

if __name__=='__main__':
    batch_size = 100
    train_data_batched, test_data_batched = load_data_mnist(batch_size=batch_size)
    print('len(train_data_batched): ', len(train_data_batched))
    print('len(test_data_batched): ', len(test_data_batched))
    print('type(train_data_batched): ', type(train_data_batched))
    print('type(test_data_batched): ', type(test_data_batched))

    lenet = nn.Sequential()
    lenet.add(nn.Conv2D(channels=30, kernel_size=5, activation='relu'),
            nn.Conv2D(channels=30, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(100, activation='relu'),
            nn.Dense(10))
    lr = 0.1
    num_epochs = 10
    lenet.initialize(force_reinit=True, init=init.Normal(), ctx=ctx)
    trainer = gluon.Trainer(lenet.collect_params(), 'sgd', {'learning_rate': lr})
    do_train(net=lenet, 
        train_iter=train_data_batched, test_iter=test_data_batched, 
        batch_size=batch_size, trainer=trainer, 
        num_epochs=num_epochs, ctx=ctx)


    






    
