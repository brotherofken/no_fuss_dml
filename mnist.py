#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc

import theano
from theano import tensor as T
import lasagne

import matplotlib.pyplot as plt

import lasagne
from lasagne.utils import floatX

import gzip

import matplotlib.pyplot as plt

from misc import nca_loss

# Download MNIST
#!wget -P data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
#!wget -P data  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# In[]
class Dataset:
    def __init__(self, path=''):
        self.train_classes = np.array(range(10))
        self.n_classes = 10

        with gzip.open("data/train-images-idx3-ubyte.gz", 'rb') as f:
            self.X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
            self.X = self.X / floatX(256)

        with gzip.open("data/train-labels-idx1-ubyte.gz", 'rb') as f:
            self.y = np.frombuffer(f.read(), np.uint8, offset=8)

        self.X_train, self.X_val = self.X[:-10000], self.X[-10000:]
        self.y_train, self.y_val = self.y[:-10000], self.y[-10000:]


    def train_batch(self, size=32):
        selection = np.random.choice(self.X_train.shape[0], size, replace=False)
        return self.X_train[selection, :], self.y_train[selection]

    def valid_batch(self, size=1024):
        selection = np.random.choice(self.X_val.shape[0], size, replace=False)
        return self.X_val[selection, :], self.y_val[selection]

data = Dataset()
train_batch, train_labels = data.train_batch(128)
valid_set, valid_labels = data.valid_batch(128)

# In[]

def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    input_layer = network

    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.035), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.035), num_units=2, nonlinearity=lasagne.nonlinearities.linear)


    return network, input_layer

# In[]

l_in_labels = lasagne.layers.InputLayer((None,))

embedding_layer, l_in_images = build_cnn()

n_train_classes = len(data.train_classes)

init_proxies = np.random.randn(len(data.train_classes), embedding_layer.output_shape[1]).astype('float32')
proxies_layer = lasagne.layers.EmbeddingLayer(l_in_labels, input_size=n_train_classes, output_size=embedding_layer.output_shape[1], W=init_proxies)

in_images = l_in_images.input_var
in_labels = T.imatrix()

nca_loss

image_embeddings = lasagne.layers.get_output(embedding_layer)
image_embeddings_determenistic = lasagne.layers.get_output(embedding_layer, deterministic=True)

loss = nca_loss(in_labels, image_embeddings, proxies_layer)

params = lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True)

updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.001)

train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)
embedding_fn = theano.function(inputs=[in_images], outputs=image_embeddings_determenistic)

# In[] Debug train loop
i = 0

batch_size=128
train_batch, train_labels = data.train_batch(batch_size)
valid_set, valid_labels = data.valid_batch(512)

for i in range(int(2 * 50000/batch_size)):
    if i % 10 == 0:
        print("plot")
        plot_stuff(i)
    train_loss = train_fn(train_batch, [train_labels])
    valid_loss = validate_fn(train_batch, [train_labels])
    del train_batch
    del train_labels
    train_batch, train_labels = data.train_batch(batch_size)
    print('epoch {} loss: {}   {}'.format(i, train_loss, valid_loss))


# In[]
proxies = np.array(proxies_layer.W.eval())
proxies /= np.sqrt((proxies * proxies).sum(axis=1)).reshape(proxies.shape[0], 1)
plt.scatter(proxies[:,0], proxies[:,1], s=50)

# In[]
from matplotlib.pyplot import cm

def plot_stuff(save=None):
    colors=cm.rainbow(np.linspace(0,1,10))

    valid_set, valid_labels = data.valid_batch(2000)

    class_embeddings = np.array(embedding_fn(valid_set))
    plt.figure(figsize=(18,9))
    for cls, color in zip(range(10),colors):
        current_points = class_embeddings[valid_labels==cls]
        if save is not None:
            plt.title('Iteration {}'.format(save))
        plt.subplot(121).scatter(current_points[:,0], current_points[:,1],
                    c=color, #valid_labels[valid_labels==cls],
                    marker='${}$'.format(cls), s=100, linewidths=0.1, edgecolor='black')
        plt.subplot(121).set_xlim([-1, 1])
        plt.subplot(121).set_ylim([-1, 1])
        current_points /= np.sqrt((current_points * current_points).sum(axis=1)).reshape(current_points.shape[0], 1)
        plt.subplot(122).scatter(current_points[:1000,0], current_points[:1000,1],
                c=color, #valid_labels[valid_labels==cls],
                marker='${}$'.format(cls), s=100, linewidths=0.1, edgecolor='black')
    plt.legend()
    if save is not None:
        plt.savefig('pics/{}.png'.format(save))
        plt.close()
    else:
        plt.show()

plot_stuff()
