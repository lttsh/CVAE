import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class ConvPriorNetwork(keras.Model):
    def __init__(self, hparams):
        super(PriorNetwork, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            hparams.num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.conv2 = keras.layers.Conv2D(
            hparams.num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.mu = keras.layers.Dense(hparams.latent_dim, activation=None)
        self.sigma = keras.layers.Dense(hparams.latent_dim, activation=None)

    def call(self, x):
        x = self.conv1(x)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)
        x = self.conv2(x)
        x = keras.layers.Flatten()(x)
        return self.mu(x), self.sigma(x)

class ConvPosteriorNetwork(keras.Model):
    def __init__(self, hparams):
        super(PosteriorNetwork, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            hparams.num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.conv2 = keras.layers.Conv2D(
            hparams.num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.mu = keras.layers.Dense(hparams.latent_dim, activation=None)
        self.sigma = keras.layers.Dense(hparams.latent_dim, activation=None)

    def call(self, x):
        x, y = x
        x = keras.layers.Concatenate(axis=-1)([x, y])
        x = self.conv1(x)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)
        x = self.conv2(x)
        x = keras.layers.Flatten()(x)
        return self.mu(x), self.sigma(x)

class ConvGenerationNetwork(keras.Model):
    def __init__(self, hparams):
        super(GenerationNetwork, self).__init__()
        self.d1 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.d2 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.d3 = keras.layers.Dense(np.prod(hparams.target_size), activation=None)

    def call(self, x):
        x, z = x
        x = keras.layers.Flatten()(x)
        z = keras.layers.Flatten()(z)
        x = keras.layers.Concatenate(axis=-1)([x, z])
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
