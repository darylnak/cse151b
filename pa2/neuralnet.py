################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = x
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        z = self.sigmoid(self.x)
        return z*(1-z)

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        z = self.tanh(self.x)
        return 1-np.power(z, 2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        z = self.ReLU(self.x)
        return (z > 0).astype('float')


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x@self.w + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_w = self.x.T@delta / delta.shape[0] # get weight correction at current layer
        self.d_b = np.mean(delta, axis=0)
        self.d_x = np.sum(self.w@delta.T, axis=1) # propogate error to next layer
        
        return self.d_x
    
    def update(self, lr):
        """
        Update the weights of this layer with learning rate lr.
        """
        self.w += lr*self.d_w
        self.b += lr*self.d_b
        
        return None

class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.lr = config["learning_rate"]
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets
        out = self.x
        loss = None
        
        for layer in self.layers:
            out = layer.forward(out)
        
        self.y = self.softmax(out)
        
        if targets is not None:
            loss = self.loss(self.y, targets)
            
        return self.y, loss

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        out = self.targets - self.y
        
        # See 'Using Negative Step and Reversed List' below:
        # https://tinyurl.com/yyjc35sy
        for layer in self.layers[::-1]:
            out = layer.backward(out)
            
        return None

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        
        https://stats.stackexchange.com/questions/304758/softmax-overflow
        """
        soft = np.exp(x-np.max(x))
        soft = soft / np.sum(soft, axis=1).reshape((-1, 1))
        
        return soft

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        
        """
        loss = np.sum(np.log(logits)*targets) / targets.shape[0]
        
        return -loss
    
    def update(self):
        """
        Update the weights for each layer.
        """
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.update(self.lr)
        
        return None