#2017/08/24
# Follows http://neuralnetworksanddeeplearning.com/chap1.html
# Uses Modified NIST data set (http://yann.lecun.com/exdb/mnist/)
# Code pulled from https://github.com/mnielsen/neural-networks-and-deep-learning.git
# def journal__20170824__testitout():

import mnist_loader
import network

# Network params
nPixels=784
nHidden=30
nEpochs=30

# Params for stochastic gradient descent (SGD)
miniBatchSize=10
learningRate=3.0

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Setup network
net = network.Network([nPixels, nHidden, miniBatchSize])

# Train!
net.SGD(training_data, nEpochs, miniBatchSize, learningRate, test_data=test_data)
