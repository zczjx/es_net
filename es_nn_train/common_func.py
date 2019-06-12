#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def identity_equal(x):
    return x

def softmax(x):
    max_bia = np.max(x)
    exp_arr = np.exp(x - max_bia)
    sum_exp_arr = np.sum(exp_arr)
    y = exp_arr / sum_exp_arr
    return y

def square_sum_func(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(func, x):
    dx = 1e-4
    grad_arr = np.zeros_like(x)

    for idx in range(x.size):
        xval_save = x[idx]
        x[idx] = xval_save + dx
        fdx1 = func(x)

        x[idx] = xval_save - dx
        fdx2 = func(x)

        grad_arr[idx] = (fdx1 - fdx2) / (2 * dx)
        x[idx] = xval_save

    return grad_arr

def gradient_descent(func, init_x, learning_rate=0.01, step_num=100):
    x = init_x
    x_trace = []

    for i in range(step_num):
        x_trace.append(x.copy())
        grad = numerical_gradient(func=func, x=x)
        x -= learning_rate * grad
    
    return x, np.array(x_trace)



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

