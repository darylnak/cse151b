################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from utils import load_data, load_config, write_to_file, one_hot_encoding, split_data, normalize
from train import *

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    # data is shape num_imgs X width X height
    x_train, y_train, x_test, y_test = load_data()

    # Create validation set out of training data.
    x_train, y_train, x_val, y_val = split_data(x_train, y_train)
    x_test = x_test.reshape((x_test.shape[0], -1)) # num X features

    # Any pre-processing on the datasets goes here.
    x_train, x_val, x_test = normalize(x_train, x_val, x_test)
    y_train = one_hot_encoding(y_train)
    y_val   = one_hot_encoding(y_val)
    y_test  = one_hot_encoding(y_test)
    
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)
