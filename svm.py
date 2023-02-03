import os
import sys
import argparse
import numpy as np
from datetime import datetime

from mnist_reader import load_mnist
import utils
from pegasos import PegasosSVM
from kernelized_pegasos import KernelPegasosSVM

def parse_arguments():
    # args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--classes', type = list, required=False, default=[1, 2], nargs='+')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--kernel', type=str, default=False)
    parser.add_argument('--lambda', default=1, type=float)
    return parser.parse_args()

def main():
    args = parse_arguments()
    X_train, y_train = load_mnist(args.dataset_dir, kind = 'train')
    X_test, y_test = load_mnist(args.dataset_dir, kind = 'test')

    start=datetime.now()
    print("Classes: ",np.array(args.classes).squeeze().astype(int))
    X_train_1 , y_train_1, X_test_1, y_test_1 = utils.getClasses(X_train, y_train, X_test, y_test,input_size=10, classes=np.array(args.classes).squeeze().astype(int))
    
    print("Training Pegasos SVM...")
    if args.kernel=="rbf":
        psvm = KernelPegasosSVM()
        psvm.kernel_type = "rbf"
        psvm.iter = args.iterations
        psvm.fit(X_train_1, y_train_1)
        print('Accuracy for RBF SVM',psvm.classify(X_test_1, y_test_1))
    elif args.kernel=="poly":
        psvm = KernelPegasosSVM()
        psvm.kernel_type = "poly"
        psvm.iter = args.iterations
        psvm.fit(X_train_1, y_train_1) 
        print('Accuracy for poly SVM',psvm.classify(X_test_1, y_test_1)) 
    elif args.kernel=="sigmoid":
        psvm = KernelPegasosSVM()
        psvm.kernel_type = "sigmoid"
        psvm.iter = args.iterations
        psvm.fit(X_train_1, y_train_1)
        print('Accuracy for sigmoid SVM',psvm.classify(X_test_1, y_test_1))
    else:
        psvm = PegasosSVM()
        psvm.iter = args.iterations
        psvm.fit(X_train_1, y_train_1)
        print('Pegasos Linear Classifier - accuracy', psvm.classify(X_test_1,y_test_1))

    print('Time taken', datetime.now()-start)

main()

