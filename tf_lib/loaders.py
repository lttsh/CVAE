import os
import numpy as np
import tensorflow as tf
from tf_lib.datasets import prepare_image_dataset

def load_tinyImageNet(num_val=None, num_labels=None, preproc=None, **kwargs):
    test_folder = os.path.join('dataset', 'tiny-imagenet-200', 'test', 'images')
    test_images = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    images = []
    for f in test_images:
        image = Image.open(os.path.join(test_folder, f))
        image = image.resize((32, 32))
        image = np.array(image)
        if len(image.shape) < 3:
            image = np.stack([image, image, image], axis=-1)

        images.append(image)
    images = np.stack(images,axis=0)
    print(images.shape)
    if preproc is not None:
        images = preproc(images)
    return [images, np.zeros(len(images)), np.ones(len(images))]

def load_gaussian_noise(sample_size, mean=0., variance=1., **kwargs):
    samples = np.random.RandomState(seed=GAUSSIAN_SEED).randn(*sample_size) * np.sqrt(variance) + mean
    return [samples, np.zeros(sample_size[0]), np.ones(sample_size[0])]

def load_svhn(num_val=None, num_labels=None, preproc=None, **kwargs):
    train_x = scipy.io.loadmat('dataset/svhn/train_32x32.mat')
    test_x = scipy.io.loadmat('dataset/svhn/test_32x32.mat')

    train_x, train_y = train_x['X'], train_x['y']
    test_x, test_y = test_x['X'], test_x['y']
    train_x = np.transpose(train_x, (3, 0, 1, 2))
    test_x = np.transpose(test_x, (3, 0, 1, 2))
    return prepare_image_dataset(train_x, train_y, test_x, test_y, num_val, num_labels, preproc)

def load_cifar100(num_val=None, num_labels=None, preproc=None, **kwargs):
    '''
    validation_prop: None means no validation split creates random validation set.
    num_labeled: None (all are labeled), number of labels per class on the train set, assigned randomly
    preproc: function that preprocesses data
    returns iterators to all splits.
    '''
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()

    return prepare_image_dataset(train_x, train_y, test_x, test_y, num_val, num_labels, preproc)

def load_mnist(num_val=None, num_labels=None, preproc=None, **kwargs):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    return prepare_image_dataset(train_x, train_y, test_x, test_y, num_val, num_labels, preproc)
