import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class PriorNetwork(keras.Model):
    def __init__(self, hparams):
        super(PriorNetwork, self).__init__()
        self.d1 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.d2 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.mu = keras.layers.Dense(hparams.latent_dim, activation=None)
        self.sigma = keras.layers.Dense(hparams.latent_dim, activation=None)

    def call(self, x):
        x = keras.layers.Flatten()(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.mu(x), self.sigma(x)

class PosteriorNetwork(keras.Model):
    def __init__(self, hparams):
        super(PosteriorNetwork, self).__init__()
        self.d1 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.d2 = keras.layers.Dense(hparams.num_hidden, activation=tf.nn.relu)
        self.mu = keras.layers.Dense(hparams.latent_dim, activation=None)
        self.sigma = keras.layers.Dense(hparams.latent_dim, activation=None)

    def call(self, x):
        x, y = x
        x = keras.layers.Flatten()(x)
        y = keras.layers.Flatten()(y)
        x = keras.layers.Concatenate(axis=-1)([x, y])
        x = self.d1(x)
        x = self.d2(x)
        return self.mu(x), self.sigma(x)

class GenerationNetwork(keras.Model):
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
