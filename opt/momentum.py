from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core import InnerOptimizer


class MomentumInnerOptimizer(InnerOptimizer):
    def __init__(self, learning_rate=0.1, momentum=0.75, n_iters=10,
                 learnable_learning_rate=False,
                 learnable_momentum=False,
                 gradient_clip_value=None):
        if learnable_learning_rate:
            learning_rate = tf.Variable(learning_rate, name='learning_rate')
            tf.summary.scalar('inner_learning_rate', learning_rate)
        if learnable_momentum:
            momentum = tf.Variable(momentum, name='momentum')
            tf.summary.scalar('inner_momentum', momentum)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_iters = n_iters
        self.gradient_clip_value = gradient_clip_value

    def initial_conditions(self, f, x):
        return (tf.zeros_like(x, name='accumulation'),)

    def step(self, f, x, accumulation):
        fx = f(x)
        dfdx, = tf.gradients(fx, x)
        clip = self.gradient_clip_value
        if clip is not None:
            dfdx = tf.clip_by_value(dfdx, -clip, clip)
        accumulation = self.momentum * accumulation + dfdx
        x = x - self.learning_rate * accumulation
        return x, accumulation

    @property
    def maximum_iterations(self):
        return self.n_iters
