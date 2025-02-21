from pyclassify.utils import read_config
from pyclassify.utils import read_file
from pyclassify.classifier import kNN
import random
import time
import argparse

def shuffle_data(X, Y):
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)
    return X, Y


def split_data(data,fraction_train = 0.8, shuffle=False):
    X , Y = data
    N = len(Y)
    N_train = int(fraction_train*N)
    if shuffle == False: 
        X_train = X[:N_train]
        Y_train = Y[:N_train]
        X_test = X[N_train:]
        Y_test = Y[N_train:]
    else: 
        Xs , Ys = shuffle_data(X, Y)
        X_train = Xs[:N_train]
        Y_train = Ys[:N_train]
        X_test = Xs[N_train:]
        Y_test = Ys[N_train:]       
    data_train = (X_train,Y_train)
    data_test = (X_test,Y_test)

    return data_train,data_test

def kNN_classification(fraction_train = 0.8,shuffle=False):
    t0 = time.time()
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config')
    args = parser.parse_args()
    config_path = args.config

    # Read the configuration file
    conf = read_config(config_path)
    file = conf['dataset']
    k = conf['k']
    backhand = conf['backhand']
    # Load the data
    data = read_file(file)
    # Split the data
    data_train , data_test = split_data(data,fraction_train=fraction_train,shuffle=shuffle)
    # Initialize classifier
    classifier = kNN(k=k,backhand=backhand)
    # Unpack test data
    X_test , Y_test = data_test
    # Make predictions
    Y_prediction = classifier(data_train,X_test)
    # Count the number of mistakes
    N = len(Y_test)
    N_mistakes = sum(abs(Y_test[i]-Y_prediction[i]) for i in range(N))
    # Compute accuracy
    acc = 1.0 - N_mistakes/N
    t1 = time.time()
    dt = t1 - t0
    print(f'The number of neighbors is k = {k}')
    print(f'The accuracy of the model is {acc*100:.2f} %')
    print(f'Prediction time = {dt:.3f} seconds')


if __name__ == '__main__': 
    kNN_classification(shuffle=False)
