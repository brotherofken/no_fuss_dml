#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import scipy as sp
from scipy import ndimage
import scipy.io as sio
from scipy import misc
import pickle

import theano
from theano import tensor as T
import lasagne

import matplotlib.pyplot as plt

from misc import nca_loss

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
    def __init__(self, path='/home/rakhunzy/workspace/data/cars196'):
        self.path = path
        annotations = sio.loadmat(os.path.join(path,'cars_annos.mat'))['annotations']
        self.img_paths = [a['relative_im_path'][0] for a in annotations[0]]
        self.labels = np.array([a['class'][0][0] for a in annotations[0]]) - 1
        self.uniq_labels = np.unique(self.labels)
        self.train_classes = self.uniq_labels[:len(self.uniq_labels)//2]
        self.n_classes = len(self.uniq_labels)

    def _load_preprocess_img(self, imgpath):
        from inception_v3 import preprocess
        img = sp.ndimage.imread(os.path.join(self.path, imgpath))
        if len(img.shape)==2:
            img = np.rollaxis(np.stack([img, img, img]), 0, 3)
        return preprocess(img)

    def _gen_data(self, size, filter_fn):
        from inception_v3 import preprocess
        imgids = []
        images = []
        labels = []

        while len(images) < size:
            imgid = np.random.randint(len(self.img_paths))
            imgpath = self.img_paths[imgid]
            imgclass = self.labels[imgid]

            if not filter_fn(imgid, imgclass) and imgid not in imgids:
                continue

            img = sp.ndimage.imread(os.path.join(self.path, imgpath))

            if len(img.shape)==2:
                img = np.rollaxis(np.stack([img, img, img]), 0, 3)

            imgids.append(imgid)
            images.append(preprocess(img))
            labels.append(imgclass)
        return np.concatenate(images), np.array(labels).astype(np.int32)

    def train_batch(self, size=32):
        return self._gen_data(size, lambda i, c: c in self.train_classes)

    def valid_batch(self, size=1024):
        return self._gen_data(size, lambda i, c: c not in self.train_classes)

    def iterate_minibatches(self, batchsize, shuffle=True, train=True):
        indices = []
        if train:
            indices = np.argwhere(np.in1d(data.labels, data.train_classes))
        else:
            indices = np.argwhere(np.logical_not(np.in1d(data.labels, data.train_classes)))

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.img_paths) - batchsize + 1, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            images = [self._load_preprocess_img(self.img_paths[int(i)]) for i in excerpt]
            if len(images) == batchsize:
                yield np.concatenate(images), np.array(self.labels[excerpt]).astype(np.int32).T
            else:
                raise StopIteration

data = Dataset()

    # In[]
train_batch, train_labels = data.train_batch(32)
valid_set, valid_labels = data.valid_batch(128)

# In[]
l_in_labels = lasagne.layers.InputLayer((None,))
l_in_images = inception_net['input']

embedding_size = 64 #embedding_layer.output_shape[1]
init_proxies = np.random.randn(len(data.train_classes), embedding_size).astype('float32')

#for imgclass in data.train_classes:
#    print('Init proxy for class {}'.format(imgclass))
#    class_images = data.images_for_class(imgclass)
#    class_embeddings = compute_embedding(class_images)
#    init_proxies[imgclass-1] = class_embeddings.mean(axis=0)

n_train_classes = len(data.train_classes)
proxies_layer = lasagne.layers.EmbeddingLayer(l_in_labels, input_size=n_train_classes, output_size=embedding_size, W=init_proxies)


in_images = inception_net["input"].input_var

inception_net["input"].params

in_labels = T.imatrix()


linear_embedding_layer = lasagne.layers.DenseLayer(embedding_layer, num_units=embedding_size, nonlinearity=lasagne.nonlinearities.linear)

image_embdeddings = lasagne.layers.get_output(linear_embedding_layer)

loss = nca_loss(in_labels, image_embdeddings, proxies_layer)

params = lasagne.layers.get_all_params([proxies_layer, linear_embedding_layer, embedding_layer], trainable=True) # embedding_layer,
#params += linear_embedding_layer.get_params()

updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.001)

len([i.__dict__['type'] for i in params])

print('Function compilation')
print('    train_fn')
train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
print('    validate_fn')
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)
print('    embedding_fn')
embedding_fn = theano.function(inputs=[in_images], outputs=image_embdeddings)
print('Done')

# In[] Main train loop
train_batch_errors =[]
valiation_errors = []
last_epoch = 0

# In[]
num_epochs = 50
for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    batch_errors = []
    for batch in data.iterate_minibatches(20, shuffle=True):
        inputs, targets = batch
        batch_loss = train_fn(inputs, targets)
        batch_errors.append(batch_loss)
        train_err += batch_loss
        train_batches += 1
        if train_batches % 50 == 0:
            print("Batch {} loss: {}.".format(train_batches, batch_loss))
    train_batch_errors.append(batch_errors)
    # And a full pass over the validation data:
    val_err = 0
    val_batches = 0
    # validion set has another classes, so validate on train
    for batch in data.iterate_minibatches(64, shuffle=False, train=True):
        inputs, targets = batch
        err = validate_fn(inputs, targets)
        val_err += err
        val_batches += 1
        if train_batches % 50 == 0:
            print("{} batches done.".format(val_batches))
    valiation_errors.append(val_err / val_batches)
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    np.savez('models/cars_196_weights_{}.npz'.format(epoch), *lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True))
last_epoch = epoch

# In[]

#plt.plot(np.concatenate(train_batch_errors))
plt.plot(valiation_errors)
plt.grid()


# In[] Check proxy values
proxies = np.array(proxies_layer.W.eval())
for i in range(len(proxies)):
    print((proxies[0] - init_proxies[0]).sum())



# In[]

valid_embeddings = []
valid_labels = []

from inception_v3 import preprocess

i = 0
for imgpath, imgclass in zip(data.img_paths, data.labels):
    i += 1
    if imgclass in data.train_classes:
        continue
    #imgpath = data.imgid2image[imgid]
    label = imgclass #data.imgid2class[imgid]
    img = sp.ndimage.imread(os.path.join(data.path, imgpath))

    # fix for grayscale images
    if len(img.shape)==2:
        img = np.rollaxis(np.stack([img, img, img]), 0, 3)
    valid_embeddings.append(embedding_fn(preprocess(img)))
    valid_labels.append(label)
    if i % 10 == 0:
        print('{}/{}'.format(i, len(data.img_paths)))

# In[]
valid_embeddings = np.concatenate(valid_embeddings)
valid_labels = np.array(valid_labels)

# In[]
from sklearn import metrics
import sklearn as ski

valid_embeddings_distances = ski.metrics.pairwise.euclidean_distances(valid_embeddings, valid_embeddings)

# In[]
valid_embeddings_distances_sorted = np.argsort(valid_embeddings_distances, axis=1)

labels_retrievals = valid_labels[valid_embeddings_distances_sorted]

k=1
np.count_nonzero(np.sum(np.array([labels_retrievals[:,0] == labels_retrievals[:,i+1] for i in range(k)]), axis=0)) / np.count_nonzero(valid_labels)

# In[]
plt.matshow(valid_embeddings_distances)










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