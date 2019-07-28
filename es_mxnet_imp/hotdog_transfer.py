#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from mxnet.gluon import model_zoo
from common_mx import *


if __name__=='__main__':

    data_dir = './data'
    train_dataset = gdata.vision.ImageFolderDataset(
                        os.path.join(data_dir, 'hotdog/train'))
    test_dataset = gdata.vision.ImageFolderDataset(
                        os.path.join(data_dir, 'hotdog/test'))

    hotdogs = [train_dataset[i+32][0] for i in range(8)]
    not_hotdogs = [train_dataset[-i - 1 - 32][0] for i in range(8)]

    # images preprocess
    normalize = gdata.vision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_img_pre_param = gdata.vision.transforms.Compose([
                                gdata.vision.transforms.RandomResizedCrop(224),
                                gdata.vision.transforms.RandomFlipLeftRight(),
                                gdata.vision.transforms.ToTensor(), normalize])

    test_img_pre_param = gdata.vision.transforms.Compose([
                                gdata.vision.transforms.Resize(256),
                                gdata.vision.transforms.CenterCrop(224),
                                gdata.vision.transforms.ToTensor(), normalize])
    
    pre_trained_net = model_zoo.vision.resnet18_v2(pretrained=True)
    # print(pre_trained_net.features)
    # print(pre_trained_net.output)
    app_net = model_zoo.vision.resnet18_v2(classes=2)
    # pre_trained_net = model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
    # app_net = model_zoo.vision.mobilenet_v2_1_0(classes=2)
    app_net.features = pre_trained_net.features
    app_net.output.initialize(init.Xavier())
    app_net.output.collect_params().setattr('lr_mult', 10)
    batch_size=32
    num_epochs=20
    learning_rate = 0.01
    train_iter = gdata.DataLoader(
                    train_dataset.transform_first(train_img_pre_param),
                    batch_size, shuffle=True)
    test_iter = gdata.DataLoader(
                    test_dataset.transform_first(test_img_pre_param),
                    batch_size)
    app_net.collect_params().reset_ctx(ctx)
    app_net.hybridize()
    # loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(app_net.collect_params(), 'sgd',
                    {'learning_rate': learning_rate, 'wd': 0.001})
    # common_train(train_iter, test_iter, app_net, loss, trainer, ctx, num_epochs=num_epochs)
    test_acc_list = do_train(net=app_net, 
                        train_iter=train_iter, test_iter=test_iter, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)
    pkl_file = os.path.basename(__file__).split('.')[0] + '.pkl'
    with open(pkl_file, 'wb') as pkl_f:
        pickle.dump(test_acc_list, pkl_f)


    