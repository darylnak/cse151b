################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import pickle
import numpy as np
import yaml


def write_to_file(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    return X_train, y_train, X_test, y_test


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    one_hot  = np.zeros((labels.shape[0], num_classes), dtype=int)

    # map each label to a unique id
    mapping = {}
    for i in range(num_classes):
        mapping[i] = i
    
    for row, label in enumerate(labels):
        one_hot[row, mapping[label]] = 1
    
    return one_hot


def find_accuracy(predicted, target):
    """
    Return accuracy of predictions.
    """
    num_correct = np.sum(target*predicted)
    
    return np.sum(num_correct) / target.shape[0]

def split_data(X, y):
    """
    Description:
                Prepares the data for training by performing k mutex splits, 
                dimensionality reduction using PCA, and adding a bias term to the data. 
    Parameters:  
                X: array_like; num_imgs X width X height
                    The original dataset.
    Returns:
                x_train: array_like
                    Data to be used for training. Returned as
                    num_imgs X 32*32
                y_train: array_like
                    Labels for training data.
                x_val: array_like
                    Data to be used for validation. Returned as
                    num_val X 32*32
                y_val: array_like
                    Labels for validation data.
    """
    num_imgs = X.shape[0]
    val_size = int(num_imgs/5) # 80:20 split of train:val
    data, labels = shuffle(X, y) # shuffle data
    
    x_train = data[:num_imgs-val_size,:,:].reshape((num_imgs-val_size, -1))
    y_train = labels[:num_imgs-val_size]

    x_val   = data[num_imgs-val_size:,:,:].reshape((val_size, -1))
    y_val   = labels[num_imgs-val_size:]

    assert (x_train.shape[0]+x_val.shape[0]) == X.shape[0]
    
    return x_train, y_train, x_val, y_val

def shuffle(data, labels):
    """
    Description:
                Shuffle the data while maintaining proper labeling.
    Parameters: 
                data: array_like
                    Data to be shuffled.
                labels: array_like
                    Labels for each data point.
    Returns:
                dataShuffle: ndarray
                    The shuffled data.
                labelShuffle: ndarray
                    The proper labels for the shuffled data.
    """
    randIdxs = np.random.rand(data.shape[0]).argsort() # get random indices

    # Take elements from an array along an axis.
    dataShuffle  = np.take(data, randIdxs, axis=0)
    labelShuffle = np.take(labels, randIdxs, axis=0)
    
    return dataShuffle, labelShuffle

def normalize(train, val, test):
    """
    Description:
                Z-score normalize training and validation data.
    Parameters: 
                train: array_like
                    Training data.
                val: array_like
                    Validation data.
    Returns:
                train_norm: array_like
                    Z-score normalized training data.
                val_norm: array_like
                    Z-score normalized validation data.
    """
    # Do we use original x_train or x_train after removing validation?
    train_norm = (train - np.mean(train, axis=0))/np.std(train, axis=0)
    val_norm = (val - np.mean(train, axis=0))/np.std(train, axis=0)
    test_norm = (test - np.mean(train, axis=0))/np.std(train, axis=0)
    
    return train_norm, val_norm, test_norm
