#----------------------------------------------------------------------
# Importing useful libraries                                                           
#----------------------------------------------------------------------
import numpy as np #arry manipulation library
import tflearn #import ML library
import tflearn.datasets.mnist as mnist #import data

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d #convolution, max pooling
from tflearn.layers.estimator import regression
from scipy import misc 

#----------------------------------------------------------------------
# Loading The data                                                           
#----------------------------------------------------------------------
x,y,test_x,test_y = mnist.load_data(one_hot = True)

#----------------------------------------------------------------------
# Preprocessing the data                                                           
#----------------------------------------------------------------------
x = x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

#----------------------------------------------------------------------
# Building the Convolutional Neural Network                                                
#----------------------------------------------------------------------
network = input_data(shape=[None, 28, 28, 1],name='input') #input layer

network = conv_2d(network, nb_filter=4, filter_size=5, activation='relu')  #conv layer with 4 5x5 conv kernels and rectifier activiation
network = max_pool_2d(network, 2) #max pool subsampling layer with 2x2 sampling window 

network = conv_2d(network, nb_filter=4, filter_size=5, activation='relu')  #conv layer with 4 5x5 conv kernels and rectifier activiation
network = max_pool_2d(network, 2) #max pool subsampling layer with 2x2 sampling window 

network = fully_connected(network, 128, activation='tanh') #fully connected layer with 128 neurons and tanh activation function
network = fully_connected(network, 128, activation='tanh') #fully connected layer with 128 neurons and tanh activation function

network = fully_connected(network, 10, activation='softmax') #output layer with 10 neurons and softmax activation function

network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target') #regression layer with adam optimizer and crossentropy loss function

model = tflearn.DNN(network, tensorboard_verbose=0)

#----------------------------------------------------------------------
# Training the Convolutional Neural Network                                             
#----------------------------------------------------------------------
model.fit({'input': x}, {'target': y}, n_epoch=5, validation_set=({'input': test_x}, {'target': test_y}), show_metric=True, run_id='convnet_mnist')

#----------------------------------------------------------------------
# Testing the model with your own images (optional)                                       
#----------------------------------------------------------------------
image = misc.imread("test.png", flatten=True)  
image = image.reshape([-1, 28, 28, 1])

predict = model.predict({'input': image})

print(np.round_(predict, decimals=3))
print("prediction: " + str(np.argmax(predict)))