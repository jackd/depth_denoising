from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class InnerOptimizer(object):
    def initial_conditions(self, f, x):
        return ()

    def step(self, f, *args):
        raise NotImplementedError('Abstract method')

    @property
    def maximum_iterations(self):
        raise NotImplementedError('Abstract method')

    def converged(self, *args):
        return False

    def minimize(
            self, f, x0, back_prop=True, return_intermediate=True,
            explicit_loop=None):
        with tf.name_scope('inner_optimizer'):
            args = (x0,) + self.initial_conditions(f, x0)
            if return_intermediate or back_prop or explicit_loop:
                # less memory requirements when unrolled manually?
                solutions = []
                for i in range(self.maximum_iterations):
                    args = self.step(f, *args)
                    solutions.append(args[0])
                    if self.converged(*args):
                        break
                return solutions if return_intermediate else solutions[-1:]
            else:
                args = tf.while_loop(
                    lambda *args: tf.logical_not(self.converged(*args)),
                    lambda *args: self.step(f, *args),
                    args, back_prop=back_prop,
                    maximum_iterations=self.maximum_iterations)
                return [args[0]]


def get_inner_optimizer(key='momentum', **kwargs):
    """
    Helper function for assisting with deserializing.

    Args:
        key: string denoting type of optimizer to use. One of ['momentum'].
            'momentum': MomentumInnerOptimizer
        back_prop: whether or not the output is required to have gradients
            back-propagated.
        kwargs: passed to the relevant optimizer.

    Returns:
        A callable that maps (fn, x0) -> x_optimal.
    """
    if key == 'momentum':
        from .momentum import MomentumInnerOptimizer
        return MomentumInnerOptimizer(**kwargs)
    else:
        raise KeyError('Invalid key "%s"' % key)
