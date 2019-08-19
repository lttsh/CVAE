"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""

import math
import os
import numpy as np
import json

import imageio
import scipy.misc
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

def load_params(hparams, args, config_file):
    for (k,v) in vars(args).items():
        hparams.add_hparam(k, v)
    if config_file is not None:
        with open(config_file, 'r') as json_file:
            json_dict = json.load(json_file)
            for (k,v) in json_dict.items():
                hparams.add_hparam(k, v)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables(scope=None):
    model_vars = tf.trainable_variables(scope=scope)
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def show_variables(vars):
    slim.model_analyzer.analyze_vars(vars, print_info=True)

def visualize_images(images, path=None, num_rows=None):
    '''
    Visualizes images as a grid
    Input is (B, H, W, C) or (B, H, W)
    '''
    channel = None
    if len(images.shape) == 4:
        num_images, img_size, _, channel = images.shape
    else:
        num_images, img_size, _ = images.shape
    if num_rows is None:
        num_rows = int(math.ceil(np.sqrt(num_images)))
    num_columns = num_images // num_rows
    if num_columns * num_rows < num_images:
        num_columns += 1
    if channel is not None:
        grid_visualizer = np.empty((img_size * num_rows, img_size * num_columns, channel))
    else:
        grid_visualizer = np.empty((img_size * num_rows, img_size * num_columns))
    for n, x in enumerate(images):
        i = n // num_columns
        j = int(n%num_columns)
        grid_visualizer[i*img_size:i*img_size + img_size, j*img_size:j*img_size+img_size] = x
    if path is not None:
        imageio.imwrite(path, grid_visualizer.astype(np.uint8))
    return grid_visualizer

def visualize_histogram(true_bpd, false_bpd, path=None):
    plt.figure()
    plt.autoscale()
    plt.hist(true_bpd, 500, alpha=0.5, label='correct samples', density=True, histtype='stepfilled')
    plt.hist(false_bpd, 500, alpha=0.5, label='wrong samples', density=True, histtype='stepfilled')
    plt.legend(loc='best')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.clf()
    plt.close()
    plt.close('all')

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_to_logit(x, alpha):
    x = 0.5 * (x + 1.0)
    return logit(alpha + (1. - 2. * alpha) * x)

def safe_dir(*args):
    dir_path = os.path.join(*args)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path
