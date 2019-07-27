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


rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])
def img_preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)

def img_postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)

def extract_features(net, X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(net, content_img, image_shape, ctx):
    content_X = img_preprocess(content_img, image_shape).copyto(ctx)
    contents_Y, _ = extract_features(net, content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(net, style_img, image_shape, ctx):
    style_X = img_preprocess(style_img, image_shape).copyto(ctx)
    _, styles_Y = extract_features(net, style_X, content_layers, style_layers)
    return style_X, styles_Y

def content_loss(Y_hat, Y):
    return (Y_hat - Y).square().mean()

def gram(X):
    num_channels, n = X.shape[1], X.size // X.shape[1]
    X = X.reshape((num_channels, n))
    return nd.dot(X, X.T) / (num_channels * n)

def style_loss(Y_hat, Y):
    return (gram(Y_hat) - gram(Y)).square().mean()

def total_var_loss(Y_hat):
    return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                  (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y)]
    tv_l = total_var_loss(X) * tv_weight
    # 对所有损失求和
    l = nd.add_n(*styles_l) + nd.add_n(*contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


class image_composer(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(image_composer, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()

def transfer_init(X, lr, ctx):
    gen_img = image_composer(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    return gen_img(), trainer


if __name__=='__main__':
    if len(sys.argv) < 3:
        print("pls enter the content and style image path")
        raise SystemExit(1)

    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    image_shape = (320, 240)
    style_compose_net = nn.Sequential()
    pretrained_net_vgg19 = model_zoo.vision.vgg19(pretrained=True, ctx=ctx)

    for i in range(max(style_layers + content_layers) + 1):
        style_compose_net.add(pretrained_net_vgg19.features[i])
    
    style_compose_net.collect_params().reset_ctx(ctx=ctx)
    content_img = image.imread(sys.argv[1])
    style_img = image.imread(sys.argv[2])
    preprocess_content_img, contents_features = get_contents(net=style_compose_net,
                                                    content_img=content_img, 
                                                    image_shape=image_shape,
                                                    ctx=ctx)
    
    preprocess_style_img, style_features = get_styles(net=style_compose_net,
                                                    style_img=style_img, 
                                                    image_shape=image_shape,
                                                    ctx=ctx)
    learning_rate = 0.1

    new_com_img, trainer = transfer_init(X=preprocess_content_img,
                                        lr=learning_rate, ctx=ctx)
    for i in range(500):
        start = time.time()
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(net=style_compose_net,
                                X=new_com_img, content_layers=content_layers, 
                                style_layers=style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(X=new_com_img,
                    contents_Y_hat=contents_Y_hat, styles_Y_hat=styles_Y_hat,
                    contents_Y=contents_features, styles_Y=style_features)
        l.backward()
        trainer.step(1)
        nd.waitall()
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, nd.add_n(*contents_l).asscalar(),
                     nd.add_n(*styles_l).asscalar(), tv_l.asscalar(),
                     time.time() - start))
        if i % 200 == 0 and i != 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            print('change lr to %.1e' % trainer.learning_rate)
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_postprocess(preprocess_content_img).asnumpy())
    plt.subplot(2, 2, 2)
    plt.imshow(img_postprocess(preprocess_style_img).asnumpy())
    plt.subplot(2, 2, 3)
    plt.imshow(img_postprocess(preprocess_content_img - new_com_img).asnumpy())
    plt.subplot(2, 2, 4)
    plt.imshow(img_postprocess(new_com_img).asnumpy())
    plt.show()
    