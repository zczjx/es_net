#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from common_mx import *

class residual(nn.HybridBlock):
    
    def __init__(self, num_channels, use_1x1conv=False, strides=1,**kwargs):
        super(residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                            strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        
    '''    
    def forward(self, x):
        y = nd.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return nd.relu(y + x)
    '''

    def hybrid_forward(self, F, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        if self.conv3:
            x = self.conv3(x)

        return F.relu(y + x)
    

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.HybridSequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(residual(num_channels))
    return blk

       
if __name__=='__main__':
    if len(sys.argv) < 2:
        print("pls enter training epochs num")
        raise SystemExit(1)

    batch_size = 100
    prefix = 'cnn_' + os.path.basename(__file__).split('.')[0]
    syms = prefix + '-symbol.json'
    params = prefix + '-0000.params'
    onnx_file  = prefix + '.onnx'
    input_shape = (batch_size, 1, 28, 28)
    train_data_batched, test_data_batched = load_data_fashion_mnist(batch_size=batch_size)
    resnet = nn.HybridSequential()
    resnet.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    
    resnet.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    resnet.add(nn.GlobalAvgPool2D(), nn.Dense(10))

    
    X = nd.random.uniform(shape=(100, 1, 28, 28))
    resnet.initialize()
    for blk in resnet:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
    exit()
    
    

    lr = 0.1
    num_epochs = int(sys.argv[1])
    resnet.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
    resnet.hybridize()
    trainer = gluon.Trainer(resnet.collect_params(), 'sgd', {'learning_rate': lr})
    test_acc_list = do_train(net=resnet, 
                        train_iter=train_data_batched, test_iter=test_data_batched, 
                        batch_size=batch_size, trainer=trainer, 
                        num_epochs=num_epochs, ctx=ctx)

    
    resnet.export(prefix)

    # convert to onnx
    onnx_model_path = onnx_mxnet.export_model(syms, params, [input_shape], np.float32, onnx_file)

    # Load onnx model
    model_proto = onnx.load_model(onnx_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)
    