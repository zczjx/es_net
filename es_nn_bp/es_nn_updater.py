import numpy as np
from active_func import *
from cost_func import *
from common_func import *
import pdb

class SGD(object):

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, weight, bias, dW, db):
        weight -= self.lr * dW
        bias -= self.lr * db

class Momentum(object):
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_w = None
        self.v_b = None
    
    def update(self, weight, bias, dW, db):
        if self.v_w is None:
            self.v_w = np.zeros_like(weight)
        
        if self.v_b is None:
            self.v_b = np.zeros_like(bias)

        self.v_w = (self.momentum * self.v_w) - (self.lr * dW)
        self.v_b = (self.momentum * self.v_b) - (self.lr * db)
        weight += self.v_w
        bias += self.v_b
    
class AdaGrad(object):
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.h_w = None
        self.h_b = None
    
    def update(self, weight, bias, dW, db):
        if self.h_w is None:
            self.h_w = np.zeros_like(weight)
        
        if self.h_b is None:
            self.h_b = np.zeros_like(bias)
        
        self.h_w += dW * dW 
        self.h_b += db * db
        weight -= self.lr * dW / (np.sqrt(self.h_w) + 1e-7)
        bias -= self.lr * db / (np.sqrt(self.h_b) + 1e-7)

class Adam(object):
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m_w = None
        self.m_b = None
        self.v_w = None
        self.v_b = None
    
    def update(self, weight, bias, dW, db):
        if (self.m_w is None) or (self.v_w is None):
            self.m_w = np.zeros_like(weight)
            self.v_w = np.zeros_like(weight)
        
        if (self.m_b is None) or (self.v_b is None):
            self.m_b = np.zeros_like(bias)
            self.v_b = np.zeros_like(bias)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        self.m_w += (1 - self.beta1) * (dW - self.m_w)
        self.m_b += (1 - self.beta1) * (db - self.m_b)
        self.v_w += (1 - self.beta2) * (dW**2 - self.v_w)
        self.v_b += (1 - self.beta2) * (db**2 - self.v_b)

        weight -= lr_t * self.m_w / (np.sqrt(self.v_w) + 1e-7)
        bias -= lr_t * self.m_b / (np.sqrt(self.v_b) + 1e-7)
        

        