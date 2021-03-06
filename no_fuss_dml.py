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

params = lasagne.layers.get_all_params([proxies_layer, embedding_layer], trainable=True)

updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.001)

#[i.__dict__['type'] for i in params]

train_fn = theano.function(inputs=[in_images, in_labels], outputs=loss, updates=updates)
validate_fn = theano.function(inputs=[in_images, in_labels], outputs=loss)

# In[] Debug code goes below

#theano.config.exception_verbosity='high'
train_loss = train_fn(train_batch[:8], [train_labels[:8]-1])
valid_loss = validate_fn(train_batch[:8], [train_labels[:8]-1])
print(train_loss)

# In[]
embedding_fn = theano.function(inputs=[in_images], outputs=image_embdeddings)

# In[]

valid_embedding = embedding_fn(valid_set[:16])


# In[]

valid_embeddings = []
valid_labels = []

from inception_v3 import preprocess

i = 0
for imgid, imgclass in data.imgid2class.items():
    i += 1
    if imgclass in data.train_classes:
        continue
    imgpath = data.imgid2image[imgid]
    label = data.imgid2class[imgid]
    img = sp.ndimage.imread(imgpath)

    # fix for grayscale images
    if len(img.shape)==2:
        img = np.rollaxis(np.stack([img, img, img]), 0, 3)
    valid_embeddings.append(embedding_fn(preprocess(img)))
    valid_labels.append(label)
    if i % 10 == 0:
        print('{}/{}'.format(i, len(data.imgid2class)))

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

k=8
np.count_nonzero(np.sum(np.array([labels_retrievals[:,0] == labels_retrievals[:,i+1] for i in range(8)]), axis=0)) / np.count_nonzero(valid_labels)

# In[]
plt.matshow(valid_embeddings_distances)

# In[] Debug train loop
i = 0
for i in range(5000):
    train_batch, train_labels = data.train_batch(16)
    train_loss = train_fn(train_batch, [train_labels-1])
    valid_loss = 0 #validate_fn(valid_set[:16], [valid_labels[:16]-1])
    print('epoch {} loss: {}  {}'.format(i, train_loss, valid_loss))
    #class_embeddings = np.array(compute_embedding(class_images))
    #class_embeddings_mean = np.array(class_embeddings.mean(axis=0))
    #print(np.sum(class_embeddings_mean - init_proxies[-1]))

# In[] Check proxy values
proxies = np.array(proxies_layer.W.eval())
for i in range(len(proxies)):
    print((proxies[0] - init_proxies[0]).sum())














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