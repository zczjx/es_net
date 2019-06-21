#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from active_func import *
from common_func import *
from es_net import *
from es_nn_layer import *
from mnist import load_mnist

if __name__=='__main__':
    (train_data, train_label), (test_data, test_label) = load_mnist(flatten=False)
    weight_init_std=0.01
    image_size = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    input_dim = (1, 28, 28)

    # random init the dnn param
    dnn = es_net()

    
    filter_num = 30
    filter_size = 5
    filter_pad = 0
    filter_stride = 1
    # Conv layer
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
    print('conv_output_size: ', conv_output_size)
    print('pool_output_size: ', pool_output_size)
    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)

    print('Conv layer1 weight', dnn_weight_arr.shape)
    print('Conv layer1 bias', dnn_bias_arr.shape)

    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, filter_pad)
    dnn.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

     # Conv layer
    input_size = conv_output_size
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
    print('conv_output_size: ', conv_output_size)
    print('pool_output_size: ', pool_output_size)
    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, filter_num, filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)

    print('Conv layer2 weight', dnn_weight_arr.shape)
    print('Conv layer2 bias', dnn_bias_arr.shape)

    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, filter_pad)
    dnn.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

    

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    dnn.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    print('Affine layer weight', dnn_weight_arr.shape)
    print('Affine layer bias', dnn_bias_arr.shape)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr)
    dnn.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr)
    dnn.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    dnn.add_output_layer(layer_obj=layer_tmp)
    dnn.print_nn_layers()

    train_data_num = train_data.shape[0]
    test_data_num = test_data.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_data_num / batch_size, 1)
    training_iters = 12001

    print('train_data_num: ', train_data_num)
    print('iter_per_epoch: ', iter_per_epoch)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_n = 0

    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = train_data[sample_indices]
        train_label_batch = train_label[sample_indices]

        print('start ', i, ' training iterations')
        dnn.train(train_data_batch, train_label_batch)
        loss_val = dnn.loss(train_data_batch, train_label_batch)
        train_loss_list.append(loss_val)

        if i % iter_per_epoch == 0:
            sample_indices = np.random.choice(train_data_num, batch_size)
            train_data_batch = train_data[sample_indices]
            train_label_batch = train_label[sample_indices]
            train_acc = dnn.accuracy(train_data_batch, train_label_batch, batch_size)
            sample_indices = np.random.choice(test_data_num, batch_size)
            test_data_batch = test_data[sample_indices]
            test_label_batch = test_label[sample_indices]
            test_acc = dnn.accuracy(test_data_batch, test_label_batch, batch_size)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('finish epoch ', epochs_n)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
            epochs_n += 1
    
    # draw

    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_loss_list))
    x = np.arange(len(train_acc_list))
    # plt.plot(x, train_loss_list, label='train acc')
    plt.title('deep cnn traning')
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("training_iters")
    # plt.ylabel("loss_val")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
   # plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


    






    
