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
    data = pickle.load(open("inception_v3.pkl","rb"))
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

        # We could preload whole birds dataset to memory
        with open(os.path.join(self.path, 'image_class_labels.txt'), 'rb') as f:
            lines = [l.rstrip('\n').split(' ') for l in f.readlines()]
            self.imgid2class = {int(l[0]): int(l[1]) for l in lines}

        with open(os.path.join(self.path, 'images.txt'), 'rb') as f:
            lines = [l.rstrip('\n').split(' ') for l in f.readlines()]
            self.imgid2image = {int(l[0]):os.path.join(path, 'images', l[1]) for l in lines}

        self.n_classes = len(np.unique(np.array(self.imgid2class.values())))

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


    def train_batch(self, size=32):
        return self._gen_data(size, lambda i, c: c <= 100)

    def valid_batch(self, size=1024):
        return self._gen_data(size, lambda i, c: c > 100)

data = Dataset('/home/rakhunzy/workspace/data/CUB_200_2011')

# In[]
train_batch, train_labels = data.train_batch()
valid_set, valid_labels = data.valid_batch()

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

init_proxies = np.random.randn(data.n_classes, embedding_layer.output_shape[1]).astype('float32')
proxies_layer = lasagne.layers.EmbeddingLayer(l_in_labels, input_size=data.n_classes, output_size=embedding_layer.output_shape[1], W=init_proxies)

# In[]

in_images = inception_net["input"].input_var

inception_net["input"].params

in_labels = T.imatrix()
image_embdeddings = lasagne.layers.get_output(embedding_layer)

loss = nca_loss(in_labels, image_embdeddings, proxies_layer)

params = lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True)

updates = lasagne.updates.rmsprop(loss, params, learning_rate=1.0)

[i.__dict__['type'] for i in params]


# In[]
train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)



# In[] Debug code goes below
theano.config.exception_verbosity='high'
train_loss = train_fn(train_batch[:8], [train_labels[:8]])
valid_loss = validate_fn(train_batch[:8], [train_labels[:8]])
print(train_loss)

# In[]

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