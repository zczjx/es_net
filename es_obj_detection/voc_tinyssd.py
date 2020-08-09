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
    # func_start = time.time()
    anchors = MultiBoxPrior(feature_map=fmap, sizes=size, ratios=ratio)
    # print("MultiBoxPrior func time: %.4f sec" %(time.time() - func_start))
    # print(anchors)
    class_predicts = class_pred_func(fmap)
    bbox_predicts = bbox_pred_func(fmap)
    return(fmap, anchors, class_predicts, bbox_predicts)

class tinyssd(nn.Module):
    def __init__(self, num_classes):
        super(tinyssd, self).__init__()
        self.num_classes = num_classes
        self.num_scale = 5
        self.anchor_scales = torch.tensor([
                        [0.2, 0.272],
                        [0.37, 0.447],
                        [0.54, 0.619],
                        [0.71, 0.79],
                        [0.88, 0.961]], device=device)
        self.ratios = torch.tensor([[1, 2, 0.5]] * self.num_scale, device=device)
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
        #anchors = torch.tensor([0.] * 5, device=device)
        #class_preds = torch.tensor([0.] * 5, device=device)
        # bbox_preds = torch.tensor([0.] * 5, device=device)
        in_x = X
        for i in range(self.num_scale):
            in_x, anchors[i], class_preds[i], bbox_preds[i] = blk_forward(in_x,
                getattr(self, 'blk_%d' % i), self.anchor_scales[i], self.ratios[i],
                getattr(self, 'class_predictor_%d' % i),
                getattr(self, 'bbox_predictor_%d' % i))
        # print("type(anchors): ", type(anchors))
        # print("type(class_preds): ", type(class_preds))
        # print("type(bbox_preds): ", type(bbox_preds))
        # anchors = torch.tensor(anchors, device=device)
        # class_preds = torch.tensor(class_preds, device=device)
        # bbox_preds = torch.tensor(bbox_preds, device=device)

        all_anchors = torch.cat(tuple(anchors), dim=1)
        class_preds = concat_predict(class_preds)
        all_class_preds = class_preds.reshape((class_preds.shape[0], -1, self.num_classes + 1))
        all_bbox_preds = concat_predict(bbox_preds)
        # print("type(all_anchors): ", type(all_anchors))
        # print("all_anchors.device): ", all_anchors.device)
        # print("type(all_class_preds): ", type(all_class_preds))
        # print("all_class_preds.device): ", all_class_preds.device)
        # print("type(all_bbox_preds): ", type(all_bbox_preds))
        # print("all_bbox_preds.device): ", all_bbox_preds.device)
        return all_anchors, all_class_preds, all_bbox_preds

class_loss = torch.nn.CrossEntropyLoss()
bbox_loss = torch.nn.L1Loss()

def total_loss_func(class_preds, class_labels,
                bbox_preds, bbox_labels, bbox_masks):
    class_loss_val = 0
    for idx in range(class_preds.shape[0]):
        class_loss_val += class_loss(class_preds[idx], class_labels[idx])
    bbox_loss_val = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return class_loss_val + bbox_loss_val

def class_accuracy_eval(class_preds, class_labels):
    return (class_preds.argmax(axis=-1) == class_labels).sum()

def bbox_accuracy_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum()


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter the epochs exam: 20")
        raise SystemExit(1)

    batch_size = 32
    width = 320
    height = 240
    # train_iter, validate_iter = load_data_pikachu(batch_size, edge_size)
    voc2012_train_iter = load_vocdetection_format_dataset(batch_size=batch_size,
                                                          height=height, width=width,
                                                          image_set='train', year='2012')
    print('len(voc2012_train_iter): ', len(voc2012_train_iter))
    print('type(voc2012_train_iter): ', type(voc2012_train_iter))
    # print('voc2012_train_iter.shape: ', voc2012_train_iter.shape)
    # print('voc2012_train_iter: ', voc2012_train_iter)
    '''
    for data, labels in voc2012_train_iter:
        print('len(data): ', len(data))
        print('type(data): ', type(data))
        print('data.shape: ', data.shape)

        print('len(labels): ', len(labels))
        print('type(labels): ', type(labels))
        # for label in labels:
        #     print('label.shape: ', label.shape)

    exit(1)
    '''
    ssd_net = tinyssd(num_classes=len(voc_classes))
    # ssd_net.eval()
    '''
    X = torch.rand((batch_size, 3, height, width)).to(device)
    anchors, class_preds, bbox_preds = ssd_net(X)
    print('type(X)', type(X))
    print('X.shape:', X.shape)
    print('output anchors:', anchors.shape)
    print('output class preds:', class_preds.shape)
    print('output bbox preds:', bbox_preds.shape)
    exit(1)
    '''

    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    num_epochs = int(sys.argv[1])
    # optimizer  = torch.optim.Adam(ssd_net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(ssd_net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    ssd_net = ssd_net.to(device)
    # training
    for epoch in range(num_epochs):
    # TODO:
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        start = time.time()
        print('start epoch: %d' % epoch)
        iter_cnt = 0
        for data, labels in voc2012_train_iter:
            iter_cnt += 1
            iter_start = time.time()
            X = data.cuda()
            anchors, cls_preds, bbox_preds = ssd_net(X)
            bbox_labels, bbox_masks, cls_labels = MultiBoxTarget(anchors,
                                                                label_list=labels)
            # 根据类别和偏移量的预测和标注值计算损失函数
            loss_func = total_loss_func(cls_preds, cls_labels,
                                        bbox_preds, bbox_labels,
                                        bbox_masks)
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()
            acc_sum += class_accuracy_eval(cls_preds, cls_labels)
            n += cls_labels.numel()
            mae_sum += bbox_accuracy_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.numel()
            print("each iter time: %.4f sec" %(time.time() - iter_start))
        print("epoch time: %.2f sec" %(time.time() - start))
        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum.item() / n, mae_sum.item() / m, time.time() - start))
    print('finish training')
    torch.save(ssd_net.state_dict(), 'voc_tinyssd.pth')
    print('save model')