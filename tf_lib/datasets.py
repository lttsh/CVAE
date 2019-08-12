import numpy as np
import tensorflow as tf
import os
import scipy.io
from PIL import Image

VAL_SEED=0 # Seed for Validation split
GAUSSIAN_SEED=24354 # Seed for Gaussian noise dataset

def dataset_iterator(data, batch_size, transform=None, random=True):
    '''
    Usage:
    data: list of different components in dataset to be iterated on together. ie [images, labels, masks]
    batch_size
    Yields an iterator that gives back dataset components and permutation indices
    '''
    i = 0
    N = len(data[0])
    iters = N // batch_size
    if random:
        permutation = np.random.permutation(N)
    else:
        permutation = np.arange(N)
    while i < iters:
        batch_indices = permutation[i * batch_size: (i+1) * batch_size]
        yield [x[batch_indices] for x in data] + [batch_indices]
        i += 1
    if i * batch_size < N:
        batch_indices = permutation[i * batch_size:]
        yield [x[batch_indices] for x in data] + [batch_indices]

def drop_labels(labels, num_labeled):
    mask = np.zeros(len(labels))
    for c in range(100):
        class_indices = np.array([i for i in range(len(labels)) if labels[i]==c])
        keep_indices = np.random.RandomState(seed=LABEL_SEED).permutation(len(class_indices))
        mask[class_indices[keep_indices[:num_labeled]]] = 1
    return mask

def prepare_image_dataset(train_x, train_y, test_x, test_y, num_val=None, num_labels=None, preproc=None):
    # Apply pre-processing function on inputs
    if preproc is not None:
        train_x = preproc(train_x)
        test_x = preproc(test_x)

    test_y = np.reshape(test_y, (-1,))
    train_y = np.reshape(train_y, (-1,))

    # #Val/Train split up
    if num_val is not None:
        indices = np.random.RandomState(seed=VAL_SEED).permutation(len(train_x))
        val_x = train_x[indices[:num_val]]
        val_y = train_y[indices[:num_val]]
        train_x = train_x[indices[num_val:]]
        train_y = train_y[indices[num_val:]]
        val_mask = np.ones(len(val_x))
        val_data = [val_x, val_y, val_mask]
    else:
        val_data = None

    test_mask = np.ones(len(test_x))
    test_data = [test_x, test_y, test_mask]

    train_data = [train_x, train_y]
    train_mask = np.ones(len(train_x))

    if num_labels is not None:
        train_mask = drop_labels(train_y, num_labels)
    train_data.append(train_mask)

    if val_data is None:
        val_data = test_data
    print("[*] Created dataset: train: {} ({} labeled), val: {}, test: {}".format(
        len(train_x), np.sum(train_mask), len(val_data[0]), len(test_x)
    ))
    return train_data, val_data, test_data

if __name__ == '__main__':
    from utils import visualize_images
    from processing import *
    def preproc(x):
        return recenter(x)
    train_it, val_it, test_it = load_cifar100(num_val=5000, num_labels=100, preproc=preproc)
    x = next(dataset_iterator(train_it, 64))
    data = x[0]
    visualize_images(padding(data, 20), path='padded.jpg')

    train_it, val_it, test_it = load_svhn(num_val=5000, num_labels=None, preproc=preproc)
    x = next(dataset_iterator(train_it, 64))
    data = x[0]
    visualize_images(padding(data, 20), path='svhn.jpg')
