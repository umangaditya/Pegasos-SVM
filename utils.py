import numpy as np

# print(X_test)
def normalize_classes(y, minlab, maxlab):
    return [int(2*((ele - minlab)/(maxlab - minlab)) -1) for ele in y]

def getClasses(X_train, y_train, X_test, y_test,classes = [1,2], input_size=1):
    print("Preparing Dataset...")
    smaller_training_X = []
    smaller_training_Y = []
    smaller_testing_X = []
    smaller_testing_Y = []  
    for i in range(len((X_train))):
        if y_train[i] in classes:
            smaller_training_X.append(X_train[i])
            smaller_training_Y.append(y_train[i])
    for i in range(len(X_test)):
        if y_test[i] in classes:
            smaller_testing_X.append(X_test[i])
            smaller_testing_Y.append(y_test[i])
    smaller_training_Y = normalize_classes(smaller_training_Y, min(classes), max(classes))
    smaller_testing_Y = normalize_classes(smaller_testing_Y, min(classes), max(classes)) 
    return np.array(smaller_training_X[0:int(len(smaller_training_X)*input_size)]), np.array(smaller_training_Y[0:int(len(smaller_training_Y)*input_size)]), np.array(smaller_testing_X), np.array(smaller_testing_Y)
