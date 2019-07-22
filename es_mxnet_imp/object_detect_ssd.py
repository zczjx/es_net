#!/usr/bin/env python3
# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

def class_predicor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predicor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

def concat_pred(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(num_chs):
    blk = nn.HybridSequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_chs, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_chs),
                nn.Activation('relu'))
    
    blk.add(nn.MaxPool2D(2))
    return blk

def base_net():
    blk = nn.HybridSequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk

def blk_forward(X, blk, size, ratio, class_pred_func, bbox_pred_func):
    fmap = blk(X)
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=size, ratios=ratio)
    class_preds = class_pred_func(fmap)
    bbox_preds = bbox_pred_func(fmap)
    return(fmap, anchors, class_preds, bbox_preds)

class obj_ssd(nn.HybridBlock):
    def __init__(self, num_classes, import_sym=False, prefix=None, **kwargs):
        super(obj_ssd, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                        [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        if import_sym == False:
            for i in range(5):
                setattr(self, 'blk_%d' % i, get_blk(i))
                setattr(self, 'class_pred_func_%d' % i, class_predicor(self.num_anchors, 
                                                                self.num_classes))
                setattr(self, 'bbox_pred_func_%d' % i, bbox_predicor(self.num_anchors))
        else:
            X = mx.nd.ones((1,3,256,256), ctx=ctx)
            for i in range(4):
                print('init sysm blk %d' % i)
                sym_filename = prefix + '_blk_' + str(i) + '-symbol.json'
                param_filename = prefix + '_blk_' + str(i) + '-0000.params'
                net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                            param_filename, ctx=ctx)
                X = net(X)
                fmap = X
                setattr(self, 'blk_%d' % i, net)

                sym_filename = prefix + '_class_pred_func_' + str(i) + '-symbol.json'
                param_filename = prefix + '_class_pred_func_' + str(i) + '-0000.params'
                net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                            param_filename, ctx=ctx)
                net(fmap)
                setattr(self, 'class_pred_func_%d' % i, net)

                sym_filename = prefix + '_bbox_pred_func_' + str(i) + '-symbol.json'
                param_filename = prefix + '_bbox_pred_func_' + str(i) + '-0000.params'
                net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                            param_filename, ctx=ctx)
                net(fmap)
                setattr(self, 'bbox_pred_func_%d' % i, net)

            i += 1
            sym_filename = prefix + '_blk_' + str(i) + '-symbol.json'
            param_filename = prefix + '_blk_' + str(i) + '-0000.params'
            net = nn.SymbolBlock.imports(sym_filename, ['data'], ctx=ctx)
            print('init sysm blk %d' % i)
            X = net(X)
            fmap = X
            setattr(self, 'blk_%d' % i, net)

            sym_filename = prefix + '_class_pred_func_' + str(i) + '-symbol.json'
            param_filename = prefix + '_class_pred_func_' + str(i) + '-0000.params'
            net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                            param_filename, ctx=ctx)
            net(fmap)
            setattr(self, 'class_pred_func_%d' % i, net)
            sym_filename = prefix + '_bbox_pred_func_' + str(i) + '-symbol.json'
            param_filename = prefix + '_bbox_pred_func_' + str(i) + '-0000.params'
            net = nn.SymbolBlock.imports(sym_filename, ['data'],
                                            param_filename, ctx=ctx)
            net(fmap)
            setattr(self, 'bbox_pred_func_%d' % i, net)
    
    def forward(self, X):
        anchors, class_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], class_preds[i], bbox_preds[i] = blk_forward(X, 
                getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                getattr(self, 'class_pred_func_%d' % i),
                getattr(self, 'bbox_pred_func_%d' % i))
        
        return (nd.concat(*anchors, dim=1),
                concat_pred(class_preds).reshape((0, -1, self.num_classes+1)),
                concat_pred(bbox_preds))
    
    def export(self, prefix, epoch=0):
        X = mx.nd.ones((1,3,256,256), ctx=ctx)
        for i in range(5):
            net = getattr(self, 'blk_%d' % i)
            net.hybridize()
            X = net(X)
            file_prefix = prefix + '_blk_' + str(i)
            net.export(file_prefix, epoch=epoch)
            fmap = X

            net = getattr(self, 'class_pred_func_%d' % i)
            net.hybridize()
            net(fmap)
            file_prefix = prefix + '_class_pred_func_' + str(i)
            net.export(file_prefix, epoch=epoch)

            
            net = getattr(self, 'bbox_pred_func_%d' % i)
            net.hybridize()
            net(fmap)
            file_prefix = prefix + '_bbox_pred_func_' + str(i)
            net.export(file_prefix, epoch=epoch)



        # ssd_net.hybridize()
        # ssd_net.export('zcz_ssd_net')
    

class_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()

def total_loss_func(class_preds, class_labels, bbox_preds, bbox_labels, bbox_masks):
    class_loss_val = class_loss(class_preds, class_labels)
    bbox_loss_val = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return class_loss_val + bbox_loss_val

def class_accuracy_eval(class_preds, class_labels):
    return (class_preds.argmax(axis=-1) == class_labels).sum().asscalar()

def bbox_accuracy_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

def inference(X, net):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]

def display(img, output, threshold):
    fig = plt.imshow(img.asnumpy())
    # fig = plt.imshow(img)
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'r')
    plt.show()

if __name__=='__main__':

    batch_size, edge_size = 16, 256
    train_iter, validate_iter = load_data_pikachu(batch_size, edge_size)

    ssd_net = obj_ssd(num_classes=1)

    '''
    ssd_net.initialize()
    X = nd.zeros((batch_size, 3, edge_size, edge_size))
    anchors, class_preds, bbox_preds = ssd_net(X)

    print('output anchors:', anchors.shape)
    print('output class preds:', class_preds.shape)
    print('output bbox preds:', bbox_preds.shape)
    '''
    ssd_net.initialize(init=init.Xavier(), ctx=ctx)
    # ssd_net.export(prefix='zcz_ssd')
    trainer = gluon.Trainer(ssd_net.collect_params(), 'sgd',
                            {'learning_rate': 0.2, 'wd': 5e-4})

    # training
    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # 从头读取数据
        start = time.time()
        print('start epoch: %d' % epoch)
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = ssd_net(X)
                # 为每个锚框标注类别和偏移量
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                                    anchors, Y, cls_preds.transpose((0, 2, 1)))
                # 根据类别和偏移量的预测和标注值计算损失函数
                loss_func = total_loss_func(cls_preds, cls_labels, 
                                        bbox_preds, bbox_labels,
                                        bbox_masks)
            loss_func.backward()
            trainer.step(batch_size)
            acc_sum += class_accuracy_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_accuracy_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
    print('finish training')
    test_img = image.imread('./img/pikachu.jpg')
    feature = image.imresize(test_img, edge_size, edge_size).astype('float32')
    in_val = feature.transpose((2, 0, 1)).expand_dims(axis=0)
    out_val = inference(X=in_val, net=ssd_net)
    ssd_net.export(prefix='zcz_ssd')
    display(img=test_img, output=out_val, threshold=0.3)



   
