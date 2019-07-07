#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, nd
import numpy as np

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()

def identity_equal(x):
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - nd.max(x, axis=0)
        y = nd.exp(x) / nd.sum(nd.exp(x), axis=0)
        return y.T 

    x = x - nd.max(x) # avoid overflow
    return nd.exp(x) / nd.sum(nd.exp(x))

def square_sum_func(x):
    return x[0]**2 + x[1]**2

def gradient_descent(func, init_x, learning_rate=0.01, step_num=100):
    x = init_x
    x_trace = []
    x.attach_grad()

    for i in range(step_num):
        x_trace.append(x.asnumpy().copy())
        with autograd.record():
            y = func(x)
        y.backward()
        x -= learning_rate * x.grad

    return x, nd.array(x_trace, ctx=ctx)

def im2col(indut_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    indut_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = indut_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = nd.pad(indut_data, mode='constant',
                pad_width=(0, 0, 0, 0, pad, pad, pad, pad))
    img_np = img.asnumpy()
    col_np = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col_np[:, :, y, x, :, :] = img_np[:, :, y:y_max:stride, x:x_max:stride]

    col = nd.array(col_np, ctx=ctx)
    col = col.transpose(axes=(0, 4, 5, 1, 2, 3)).reshape(N*out_h*out_w, -1)
    
    return col

def col2im(col, indut_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    indut_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = indut_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(axes=(0, 3, 4, 5, 1, 2))
    col_np = col.asnumpy()

    # img = nd.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), ctx=ctx)
    img_np = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img_np[:, :, y:y_max:stride, x:x_max:stride] += col_np[:, :, y, x, :, :]

    img = nd.array(img_np, ctx=ctx)

    return img[:, :, pad:H + pad, pad:W + pad]

def weight_init_scale(input_size, active_func='relu'):

    scale = 0.01
    if str(active_func).lower() == 'relu':
        print('using He init method')
        scale = np.sqrt(2.0 / input_size)
    
    elif str(active_func).lower() == 'sigmoid':
        print('using Xavier init method')
        scale = np.sqrt(1.0 / input_size)
    
    else:
        print('using Default init method')
        
    return scale



if __name__=='__main__':
    init_x = nd.array([-5.0, 4.0], ctx=ctx)
    x, xtrace = gradient_descent(func=square_sum_func, init_x=init_x, 
                    learning_rate=0.1, step_num=40)

    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(xtrace[:,0].asnumpy(), xtrace[:,1].asnumpy(), 'o')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("X0")
    plt.ylabel("X1")

    fig = plt.figure()
    ax = Axes3D(fig)
    xval = nd.arange(-5, 5, 0.1, ctx=ctx)
    yval = nd.arange(-5, 5, 0.1, ctx=ctx)
    x, y = np.meshgrid(xval.asnumpy(), yval.asnumpy())
    z = x**2 + 0.5 * y**3
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='winter')
    plt.show()

