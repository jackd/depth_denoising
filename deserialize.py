"""Functions to help deserializing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .ops import softplus, softabs, leaky_softplus


def deserialize_activation_spec(key=None, **kwargs):
    if key is None:
        return lambda x: x
    elif key == 'relu':
        return tf.nn.relu
    elif key == 'softplus':
        kwargs.setdefault('beta', 0.04)
        return lambda x: softplus(x, **kwargs)
    elif key == 'leaky_softplus':
        return lambda x: leaky_softplus(x, **kwargs)
    elif key == 'tanh':
        return tf.nn.tanh
    elif key == 'softabs':
        kwargs.setdefault('beta', 0.04)
        return lambda x: softabs(x, **kwargs)
    elif key == 'abs':
        return tf.abs
    elif key == 'none':
        return None
    else:
        raise KeyError('Unrecognized activation key "%s"' % key)


def deserialize_initializer(stddev):
    if stddev is None:
        return None
    else:
        return tf.random_normal_initializer(stddev=stddev)


def deserialize_conv_spec(
        filters, activation_spec=None, kernel_size=7, use_bias=True,
        kernel_stddev=None):
    kernel_initializer = deserialize_initializer(kernel_stddev)
    if activation_spec is None:
        activation = None
    else:
        activation = deserialize_activation_spec(**activation_spec)
    return tf.layers.Conv2D(
        filters, kernel_size, activation=activation, use_bias=use_bias,
        kernel_initializer=kernel_initializer)


def deserialize_dense_spec(
        units, activation_spec=None, kernel_stddev=None, **kwargs):
    kernel_initializer = deserialize_initializer(kernel_stddev)
    if activation_spec is None:
        activation = None
    else:
        activation = deserialize_activation_spec(**activation_spec)
    return tf.layers.Dense(
        units, activation=activation,
        kernel_initializer=kernel_initializer, **kwargs)
