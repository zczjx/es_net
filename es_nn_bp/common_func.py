#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def identity_equal(x):
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # avoid overflow
    return np.exp(x) / np.sum(np.exp(x))

def square_sum_func(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(func, x):
    dx = 1e-4
    grad_arr = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        xval_save = x[idx]
        x[idx] = xval_save + dx
        fdx1 = func(x)

        x[idx] = xval_save - dx
        fdx2 = func(x)

        grad_arr[idx] = (fdx1 - fdx2) / (2 * dx)
        x[idx] = xval_save
        it.iternext()

    return grad_arr

def gradient_descent(func, init_x, learning_rate=0.01, step_num=100):
    x = init_x
    x_trace = []

    for i in range(step_num):
        x_trace.append(x.copy())
        grad = numerical_gradient(func=func, x=x)
        x -= learning_rate * grad
    
    return x, np.array(x_trace)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



if __name__=='__main__':
    init_x = np.array([-5.0, 4.0])
    x, xtrace = gradient_descent(func=square_sum_func, init_x=init_x, 
                    learning_rate=0.1, step_num=40)

    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(xtrace[:,0], xtrace[:,1], 'o')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("X0")
    plt.ylabel("X1")

    fig = plt.figure()
    ax = Axes3D(fig)
    xval = np.arange(-5, 5, 0.1)
    yval = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(xval, yval)
    z = x**2 + 0.5 * y**3
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='winter')
    plt.show()

