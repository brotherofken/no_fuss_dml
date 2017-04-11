# In[]
import os

import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
import pickle

import theano
from theano import tensor as T
import lasagne

import matplotlib.pyplot as plt

# In[]
def load_inception_v3():
    from inception_v3 import build_network
    net = build_network()
    data = pickle.load(open('inception_v3.pkl', 'rb'), encoding='latin1')
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["softmax"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1).astype("float32")

def compile_model_features():
    net, mean = load_inception_v3()
    output_layer = net["pool3"]
    #Y = T.ivector('y')
    X = net["input"].input_var
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)
    compute_feature = theano.function(inputs=[X], outputs=output_test)
    return net, output_layer, mean, compute_feature

# In[]
inception_net, embedding_layer, mean, compute_embedding = compile_model_features()

# In[]
# generalize to other datasets
class Dataset:
    def __init__(self, path):
        self.path = path

        self.train_classes = np.array(range(100)) + 1

        # We could preload whole birds dataset to memory
        with open(os.path.join(self.path, 'image_class_labels.txt'), 'r') as f:
            lines = [l.rstrip('\n').split(' ') for l in f.readlines()]
            self.imgid2class = {int(l[0]): int(l[1]) for l in lines}

        with open(os.path.join(self.path, 'images.txt'), 'r') as f:
            lines = [l.rstrip('\n').split(' ') for l in f.readlines()]
            self.imgid2image = {int(l[0]):os.path.join(path, 'images', l[1]) for l in lines}

        self.n_classes = len(np.unique(np.array(list(self.imgid2class.values()))))

    def _gen_data(self, size, filter_fn):
        from inception_v3 import preprocess
        imgids = []
        images = []
        labels = []

        while len(images) < size:
            imgid = np.random.randint(len(self.imgid2image)) + 1
            imgpath = self.imgid2image[imgid]
            imgclass = self.imgid2class[imgid]
            img = sp.ndimage.imread(imgpath)

            # fix for grayscale images
            if len(img.shape)==2:
                img = np.rollaxis(np.stack([img, img, img]), 0, 3)

            if filter_fn(imgid, imgclass):
                imgids.append(imgid)
                images.append(preprocess(img))
                labels.append(imgclass)
        return np.concatenate(images), np.array(labels).astype(np.int32)

    def images_for_class(self, cid):
        from inception_v3 import preprocess
        images = []

        for imgid, imgclass in self.imgid2class.items():
            if imgclass != cid:
                continue
            imgpath = self.imgid2image[imgid]
            img = sp.ndimage.imread(imgpath)

            # fix for grayscale images
            if len(img.shape)==2:
                img = np.rollaxis(np.stack([img, img, img]), 0, 3)

            images.append(preprocess(img))

        return np.vstack(images)

    def train_batch(self, size=32):
        return self._gen_data(size, lambda i, c: c in self.train_classes)

    def valid_batch(self, size=1024):
        return self._gen_data(size, lambda i, c: c not in self.train_classes)


data = Dataset('/home/rakhunzy/workspace/data/CUB_200_2011')

# In[]
train_batch, train_labels = data.train_batch(32)
valid_set, valid_labels = data.valid_batch(128)

# In[]
def nca_loss(targets, embeddings, proxies_layer):
    """Return the NCA for No Fuss DML.
    For details, we refer you to 'Neighbourhood Component Analysis' by
    J Goldberger, S Roweis, G Hinton, R Salakhutdinov (2004).
    Parameters
    ----------
    targets : Theano variable
        An array of shape ``(n,)`` where ``n`` is the number of samples. Each
        entry of the array should be an integer between ``0`` and ``k - 1``,
        where ``k`` is the number of classes.
    embeddings : Theano variable
        An array of shape ``(n, d)`` where each row represents a point in``d``-dimensional space.
    proxies_layer : Lasagne procies embedding layer
        Should contain An array of shape ``(m, d)`` where each row represents a proxy-point in``d``-dimensional space.
    Returns
    -------
    res : Theano variable
        Array of shape `(n, 1)` holding a probability that a point is
        classified correclty.
    """

    from misc import distance_matrix
    from misc import l2

    embeddings_normed = embeddings / l2(embeddings)
    # Positive distances
    target_proxies = lasagne.layers.get_output(proxies_layer, in_labels)[0]
    target_proxies_normed = target_proxies / l2(target_proxies)

    positive_distances = T.diagonal(distance_matrix(target_proxies_normed, embeddings_normed))
    numerator = T.exp(-positive_distances)

    # Negative distances
    proxies_normed = proxies_layer.W / l2(proxies_layer.W)
    all_distances = distance_matrix(embeddings_normed, proxies_normed)
    denominator = T.sum(T.exp(-all_distances), axis=1) - numerator

    # Compute ratio
    loss_vector = -T.log(numerator / denominator)
    loss = T.mean(loss_vector)
    return loss

# In[]
l_in_labels = lasagne.layers.InputLayer((None,))
l_in_images = inception_net['input']

init_proxies = np.random.randn(len(data.train_classes), embedding_layer.output_shape[1]).astype('float32')

for imgclass in data.train_classes:
    print('Init proxy for class {}'.format(imgclass))
    class_images = data.images_for_class(imgclass)
    class_embeddings = compute_embedding(class_images)
    init_proxies[imgclass-1] = class_embeddings.mean(axis=0)

# In[]
n_train_classes = len(data.train_classes)
proxies_layer = lasagne.layers.EmbeddingLayer(l_in_labels, input_size=n_train_classes, output_size=embedding_layer.output_shape[1], W=init_proxies)

# In[]

in_images = inception_net["input"].input_var

inception_net["input"].params

in_labels = T.imatrix()
image_embdeddings = lasagne.layers.get_output(embedding_layer)

loss = nca_loss(in_labels, image_embdeddings, proxies_layer)

params = lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True)

updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.0001)

#[i.__dict__['type'] for i in params]

# In[]
train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)

# In[] Debug code goes below

#theano.config.exception_verbosity='high'
train_loss = train_fn(train_batch[:8], [train_labels[:8]-1])
valid_loss = validate_fn(train_batch[:8], [train_labels[:8]-1])
print(train_loss)

# In[] Debug train loop
i = 0
for i in range(50):
    train_batch, train_labels = data.train_batch()
    train_loss = train_fn(train_batch[:8], [train_labels[:8]-1])
    print('epoch {} loss: {}'.format(i, train_loss))
    class_embeddings = np.array(compute_embedding(class_images))
    class_embeddings_mean = np.array(class_embeddings.mean(axis=0))
    print(np.sum(class_embeddings_mean - init_proxies[-1]))

# In[] Check proxy values
proxies = np.array(proxies_layer.W.eval())
for i in range(len(proxies)):
    print((proxies[0] - init_proxies[0]).sum())























# In[] MNIST

import gzip
import time
from lasagne.utils import floatX

import matplotlib.pyplot as plt

# Download
#!wget -P data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
#!wget -P data  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# In[]
def plot_mnist_sample(sample):
    plt.imshow(sample[0], cmap=plt.cm.Greys_r)
    plt.xticks([])
    plt.yticks([])

#plot_mnist_sample(X_train[np.random.randint(0, 10000)])

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
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    input_layer = network
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5), num_units=2, nonlinearity=lasagne.nonlinearities.tanh)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=10,
#            nonlinearity=lasagne.nonlinearities.softmax)

    return network, input_layer

# In[]

#l_in_images = T.tensor4('inputs')
#l_in_labels = T.ivector('targets')
l_in_labels = lasagne.layers.InputLayer((None,))
embedding_layer, l_in_images = build_cnn()

n_train_classes = len(data.train_classes)
init_proxies = np.random.randn(len(data.train_classes), embedding_layer.output_shape[1]).astype('float32')
proxies_layer = lasagne.layers.EmbeddingLayer(l_in_labels, input_size=n_train_classes, output_size=embedding_layer.output_shape[1], W=init_proxies)

## In[]
in_images = l_in_images.input_var
in_labels = T.imatrix()
image_embdeddings = lasagne.layers.get_output(embedding_layer)

loss = nca_loss(in_labels, image_embdeddings, proxies_layer)

params = lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True)

updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.001)

#[i.__dict__['type'] for i in params]

## In[]
train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)
compute_embedding = theano.function(inputs=[in_images], outputs=image_embdeddings)

# In[] Debug code goes below

#theano.config.exception_verbosity='high'
train_loss = train_fn(train_batch[:8], [train_labels[:8]])
print(train_loss)
valid_loss = validate_fn(train_batch[:8], [train_labels[:8]])
print(valid_loss)

# In[] Debug train loop
i = 0

train_batch, train_labels = data.train_batch(256)
valid_set, valid_labels = data.valid_batch(256)

for i in range(int(50000/256)):
    train_loss = train_fn(train_batch, [train_labels])
    valid_loss = validate_fn(train_batch, [train_labels])
    #if i % 10 == 0:
    train_batch, train_labels = data.train_batch(256)
    print('epoch {} loss: {}   {}'.format(i, train_loss, valid_loss))


# In[]
proxies = np.array(proxies_layer.W.eval())
proxies /= np.sqrt((proxies * proxies).sum(axis=1)).reshape(proxies.shape[0], 1)
plt.scatter(proxies[:,0], proxies[:,1], s=50)

# In[]
from matplotlib.pyplot import cm

colors=cm.rainbow(np.linspace(0,1,10))

class_embeddings = np.array(compute_embedding(valid_set))
#class_embeddings /= np.sqrt((class_embeddings * class_embeddings).sum(axis=1)).reshape(class_embeddings.shape[0], 1)
for cls, color in zip(range(10),colors):
    plt.scatter(class_embeddings[valid_labels==cls,0], class_embeddings[valid_labels==cls,1],
                c=color, #valid_labels[valid_labels==cls],
                marker='${}$'.format(cls), s=50, linewidths=0.1, edgecolor='black')

plt.legend()
plt.show()

# In[]





















# In[]
class_embeddings = np.array(compute_embedding(train_batch))

(proxies_layer.W.eval() - init_proxies).sum(axis=1)

# In[] Check proxy values
proxies = np.array(proxies_layer.W.eval())
for i in range(len(proxies)):
    print((proxies[0] - init_proxies[0]).sum())




# In[]








































# In[] Sandbox

target_proxies_normed_fn = theano.function(inputs=[in_labels], outputs=target_proxies_normed)
embeddings_normed_fn = theano.function(inputs=[in_images], outputs=embeddings_normed)
numerator_fn = theano.function(inputs=[in_images, in_labels], outputs=numerator)
denominator_fn = theano.function(inputs=[in_images, in_labels], outputs=denominator)
loss_vector_fn = theano.function(inputs=[in_images, in_labels], outputs=loss_vector)
loss_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)

# In[]
#proxies_ = get_embedding([train_labels[:8]])
target_proxies_normed_ = target_proxies_normed_fn([train_labels[:8]])
embeddings_normed_ = embeddings_normed_fn(train_batch[:8])

numerator_ = numerator_fn(train_batch[:8], [train_labels[:8]])
denominator_ = denominator_fn(train_batch[:8], [train_labels[:8]])
numerator_ / denominator_

loss_vector_ = loss_vector_fn(train_batch[:8], [train_labels[:8]])

# In[]
train_batch, train_labels = data.train_batch()
loss_ = loss_fn(train_batch[:8], [train_labels[:8]])
print(loss_)

# In[]
dists = compute_dist_matrix(train_batch[:8])

dists_ref = sp.spatial.distance.cdist(image_embdeddings_normed_, proxies_normed_)

proxies_layer.W.eval()
# In[]
#dist_xy = T.sum(T.pow(embedding_layer - proxies_layer, 2.), axis = 1)

#loss_nca = -T.log(T.exp(-dist_xy) / T.sum(T.exp(-sums_dist_xZ))))

# In[]