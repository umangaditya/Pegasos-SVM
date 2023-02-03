# Pegasos-SVM
Pegasos: Primal Estimated sub-GrAdient SOlver for SVM

## Setting up

Run `pip install -r requirements.txt` to install required packages.

## Running the code

### Arguments
- dataset_dir = dataset path 
- classes = which 2 classes to take, default = [1, 2]
- iterations = No. of iterations, default = 10
- kernel = Type of kernel, Available kernel = ["rbf", "poly", "sigmoid"]
- lamda = lamda for Pegasos Algo, default = 1

### Without kernel 
Default class : 1 2
```
python svm.py --dataset_dir data/fashion --iterations 10
```
Custom Class : 2 5
```
python svm.py --dataset_dir data/fashion --iterations 10 --classes 2 5
```

### With RBF Kernel 
```
python svm.py --dataset_dir data/fashion --iterations 10 --kernel 'rbf' 
```

### With Polynomial Kernel 
```
python svm.py --dataset_dir data/fashion --iterations 10 --kernel 'poly'
```

### With Sigmoid Kernel
```
python svm.py --dataset_dir data/fashion --iterations 10 --kernel 'sigmoid'
```
