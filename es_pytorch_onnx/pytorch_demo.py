#!/usr/bin/env python3
# coding: utf-8

import torch

if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    x = torch.rand(3, 3, device=device)
    print(x)
    print(x.shape)

    y = torch.tensor([4, 2, 3, 1], device=device)
    print(y)
    print('y.dtype: ', y.dtype)
    print('y.shape: ', y.shape)
    print('y.requires_grad: ', y.requires_grad)

    z = y.to('cpu').numpy()
    print(z)
    
   
