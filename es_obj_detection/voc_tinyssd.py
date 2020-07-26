#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, sys, pickle, getopt
sys.path.append(os.path.abspath('..'))
import torch
from torch import nn, optim
from torchvision import transforms
from es_pytorch_onnx.common_torch import *
from voc_dataset import *
from bbox_utils import *

def vgg_down_sample_blk(in_channels, out_channels):
    layers = []
    update_in_channels = in_channels
    num_convs = 2
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels=update_in_channels,
                              out_channels=out_channels,
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(inplace=True)]
        update_in_channels = out_channels

    layers += [nn.MaxPool2d(kernel_size=2)]
    return nn.Sequential(*layers)


def backbone_net():
    blk = nn.Sequential()
    layers = []
    pre_channels = 3
    for num_filters in [16, 32, 64]:
        layers += [vgg_down_sample_blk(in_channels=pre_channels, out_channels=num_filters)]
        pre_channels = num_filters
    return nn.Sequential(*layers)

def get_blk(i, num_scale=5):
    if i == 0:
        blk = backbone_net()
    elif i == 1:
        blk = vgg_down_sample_blk(in_channels=64, out_channels=128)
    elif i == num_scale -1:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = vgg_down_sample_blk(in_channels=128, out_channels=128)
    return blk

def class_predictor(in_channels, num_anchors, num_classes):
    return nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(in_channels, num_anchors):
    return nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=3, padding=1)

def flatten_predictor(predict):
    return predict.permute(0, 2, 3, 1).reshape(predict.size(0), -1)

def concat_predict(predicts):
    return torch.cat(tuple(flatten_predictor(p) for p in predicts), dim=1)


def blk_forward(X, blk, size, ratio, class_pred_func, bbox_pred_func):
    fmap = blk(X)
    anchors = MultiBoxPrior(feature_map=fmap, sizes=size, ratios=ratio)
    class_predicts = class_pred_func(fmap)
    bbox_predicts = bbox_pred_func(fmap)
    return(fmap, anchors, class_predicts, bbox_predicts)

class tinyssd(nn.Module):
    def __init__(self, num_classes):
        super(tinyssd, self).__init__()
        self.num_classes = num_classes
        self.num_scale = 5
        self.anchor_scales = [
                        [0.2, 0.272],
                        [0.37, 0.447],
                        [0.54, 0.619],
                        [0.71, 0.79],
                        [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * self.num_scale
        self.num_anchors = len(self.anchor_scales[0]) + len(self.ratios[0]) - 1
        for i in range(self.num_scale):
            if i == 0:
                setattr(self, 'blk_%d' % i, get_blk(i, num_scale=self.num_scale))
                setattr(self, 'class_predictor_%d' % i, class_predictor(64,
                                                            self.num_anchors, self.num_classes))
                setattr(self, 'bbox_predictor_%d' % i, bbox_predictor(64,
                                                            self.num_anchors))
            else:
                setattr(self, 'blk_%d' % i, get_blk(i, num_scale=self.num_scale))
                setattr(self, 'class_predictor_%d' % i, class_predictor(128,
                                                            self.num_anchors, self.num_classes))
                setattr(self, 'bbox_predictor_%d' % i, bbox_predictor(128,
                                                            self.num_anchors))


    def forward(self, X):
        anchors, class_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(self.num_scale):
            X, anchors[i], class_preds[i], bbox_preds[i] = blk_forward(X,
                getattr(self, 'blk_%d' % i), self.anchor_scales[i], self.ratios[i],
                getattr(self, 'class_predictor_%d' % i),
                getattr(self, 'bbox_predictor_%d' % i))

        all_anchors = torch.cat(tuple(anchors), dim=1)
        class_preds = concat_predict(class_preds)
        all_class_preds = class_preds.reshape((class_preds.shape[0], -1, self.num_classes + 1))
        all_bbox_preds = concat_predict(bbox_preds)
        return all_anchors, all_class_preds, all_bbox_preds

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the epochs exam: 20")
        raise SystemExit(1)

    batch_size = 1
    width = 320
    height = 240
    # train_iter, validate_iter = load_data_pikachu(batch_size, edge_size)
    voc2012_train_iter = load_vocdetection_format_dataset(height=height, width=width,
                                                          image_set='train', year='2012')
    print('len(voc2012_train_iter): ', len(voc2012_train_iter))
    print('type(voc2012_train_iter): ', type(voc2012_train_iter))

    ssd_net = tinyssd(num_classes=len(voc_classes))

    '''
    ssd_net.eval()
    X = torch.rand((batch_size, 3, height, width))
    anchors, class_preds, bbox_preds = ssd_net(X)

    print('output anchors:', anchors.shape)
    print('output class preds:', class_preds.shape)
    print('output bbox preds:', bbox_preds.shape)
    exit(1)
    '''


    lr = 0.001
    num_epochs = int(sys.argv[1])
    optimizer  = torch.optim.Adam(ssd_net.parameters(), lr=lr)

    # training
    for epoch in range(num_epochs):
    # TODO:
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        start = time.time()
        print('start epoch: %d' % epoch)
        for data, labels in voc2012_train_iter:
            print('type(data): ', type(data))
            print('data.size(): ', data.size())
            print('type(labels): ', type(labels))
            print('labels.size(): ', labels.size())
            X = data.to(device)
            Y = labels.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = ssd_net(X)
                # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = MultiBoxTarget(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            loss_func = total_loss_func(cls_preds, cls_labels,
                                        bbox_preds, bbox_labels,
                                        bbox_masks)
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()
            acc_sum += class_accuracy_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_accuracy_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
    print('finish training')