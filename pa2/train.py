################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *
from utils import find_accuracy
from copy import deepcopy
from tqdm import tqdm

def train(x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    return five things -
        training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
        best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    best_model = None
    best_loss = float('inf')
    epochs = config['epochs']
    batch_size = config['batch_size']
    nbatchs = int(x_train.shape[0]/batch_size)

    model = NeuralNetwork(config=config)
    
    for epoch in tqdm(range(epochs)):
        for i in range(0, x_train.shape[0], batch_size):
            start = i
            end = min(i+batch_size, x_train.shape[0])
            
            train_data = x_train[start:end]
            train_lbls = y_train[start:end]
            _, _ = model(train_data, train_lbls)
            
            model.backward()
            model.update()
        
        t_loss, t_acc = test(model, x_train, y_train)
        v_loss, v_acc = test(model, x_valid, y_valid)
        
        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        train_acc.append(t_acc)
        valid_acc.append(v_acc)
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_model = deepcopy(model)
            
        if epoch % 10 == 0:
            print("epoch: {} - t_loss: {} - v_loss: {} - t_acc: {} - v_acc: {}".format(epoch, t_loss, v_loss, t_acc, v_acc))

    # return train_acc, valid_acc, train_loss, valid_loss, best_model
    return train_loss, valid_loss, train_acc, valid_acc, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    # return loss, accuracy
    preds, loss = model(x_test, y_test)
    
    return loss, find_accuracy(preds, y_test)
