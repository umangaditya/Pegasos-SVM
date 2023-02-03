import numpy as np
import random
import math 

class PegasosSVM():
    """Implementation of SVM with SGD with PEGASOS Algorithm"""
    def __init__(self, iters=10, lamda=1):
        self.iter = iters
        self.lamda = lamda
        self.weight = 0 #Weight initialized
        self.S = 0 #Number of Samples  
        
    def fit(self, X, y):
        self.iter = int(len(X)/10.0)
        self.S = len(X) #Number of Samples
        self.weight = np.zeros(len(X[0]))
        for t in range(1,self.iter+1):
            index = random.randint(0,self.S-1)
            # Get index from 0 to S
            eta = 1.0/(t*self.lamda)
            dot = np.dot(self.weight, X[index])
            w_new =  (1 - eta*self.lamda)*self.weight
            if y[index]*dot < 1:
                w_new += eta*y[index]*X[index]
            w_new = min(1, 1.0/(math.sqrt(self.lamda)*np.linalg.norm(w_new)))*w_new
            self.weight = w_new
        return self.weight
    
    def get_prediction(self, X):
        return np.sign(np.dot(X, self.weight)).astype(int)
    
    def classify(self, X, y):
        count = 0
        for i in range(len(X)):
            if y[i] == np.sign(np.dot(self.weight,X[i])):
                count += 1
        return count/len(X)