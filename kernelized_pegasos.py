import numpy as np
import random

class KernelPegasosSVM():
    """Implementation of SVM with SGD with PEGASOS Algorithm"""
    def __init__(self, iters=10, lamda=1, kernel_type="rbf"):
        self.iter = iters
        self.lamda = lamda
        self.alpha = 0 #Weight initialized
        self.S = 0 #Number of Samples 
        self.weight = 0
        self.kernel_type = kernel_type
        self.X = 0
        
    def kernel_dot(self,point1, point2, gamma=1):
        if self.kernel_type == "rbf":
            diff = point1 - point2
            diff_squared = np.dot(diff,diff)
            return np.exp(-gamma*diff_squared)
        elif self.kernel_type == 'poly':
            dot_prod = np.dot(point1,point2)
            return (dot_prod+1)*(dot_prod+1)
        elif self.kernel_type == 'sigmoid':
            dot_prod = np.dot(point1, point2)
            gamma = 1/len(self.X)
            return np.tanh(gamma*dot_prod)
        
    def check(self, index, y , X, alpha_new, t):
        out = 0
        prod = y[index]/(self.lamda*t)
        for i in range(self.S):
            if i != index:
                out+= self.alpha[i]*self.kernel_dot(X[index], X[i])*y[i]
        self.alpha_new = alpha_new
        return prod*out
        
    def fit(self, X, y):
        self.iter = int(len(X)/10.0)
        self.S = len(X) #Number of Samples
        self.alpha = np.zeros(self.S)
        self.X = X
        self.y = y
        for t in range(1,self.iter+1):
            index = random.randint(0,self.S-1)
            # Get index from 0 to S
            alpha_new = self.alpha
            alpha_new[index] = 0
            if self.check(index, y, X, alpha_new, t) < 1:
                self.alpha_new[index] += 1
            else:
                self.alpha_new[index] = self.alpha_new[index]
            self.alpha = self.alpha_new
    
    def get_prediction(self, ele):
        factor = self.iter*self.lamda
        self.weight = 0
        for i in range(len(self.X)):
            self.weight += self.alpha[i]*self.y[i]*self.kernel_dot(self.X[i], ele)
        return self.weight
    
    def classify(self, X, y):
        count = 0
        for i in range(len(X)):
            if y[i] == np.sign(self.get_prediction(X[i])).astype(int):
                count += 1
        return count/len(X)
    