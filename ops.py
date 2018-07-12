from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def leaky_softplus(x, beta=1, alpha=0.2):
    with tf.name_scope('leaky_softplus'):
        return softplus(x, beta) - alpha*softplus(-x, beta)


def softplus(x, beta=1):
    with tf.name_scope('softplus'):
        if beta == 1:
            return tf.nn.softplus(x)
        else:
            return 1.0 / beta * tf.nn.softplus(x * beta)


def softabs(x, beta=1):
    with tf.name_scope('softabs'):
        return (
            softplus(x, beta) + softplus(-x, beta)) / 2


def get_psnr(predictions, labels, y_max=None):
    with tf.name_scope('psnr'):
        axis = range(1, len(predictions.shape))
        rmse = tf.sqrt(
            tf.reduce_mean(tf.square(predictions - labels), axis=axis))
        if y_max is None:
            y_max = tf.reduce_max(labels, axis=axis)
        return (20 / np.log(10.0)) * tf.log(y_max / rmse)
