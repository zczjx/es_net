#!/usr/bin/env python3
# coding: utf-8
import os, time, sys, pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    argc = len(sys.argv)
    if argc < 3:
        print("pls enter the images slice num exam: 0 7")
        raise SystemExit(1)
    
    plt.title('learning accuracy curve')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    for i in range(1, argc):
        with open(sys.argv[i], 'rb') as pkl_f:
            test_acc_list = pickle.load(pkl_f)
        label = sys.argv[i].split('.')[0]
        print('label: ', label)
        x = np.arange(len(test_acc_list))
        plt.plot(x, test_acc_list, label=label)
        plt.legend(loc='lower right')
    plt.show()


