from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel
import tensorflow as tf


_optimizers = {
    'adam': tf.train.AdamOptimizer,
}


def get_train_model(
        batch_size=64, max_steps=10000,
        optimizer_key='adam', learning_rate=1e-3,
        gradient_clip_value=None):
    if optimizer_key not in _optimizers:
        raise KeyError(
            'optimizer_key must be in %s' % str(tuple(_optimizers.keys())))

    if gradient_clip_value is None:
        def get_train_op(loss, global_step):
            optimizer = _optimizers[optimizer_key](learning_rate=learning_rate)
            return optimizer.minimize(loss, global_step)
    else:
        def get_train_op(loss, global_step):
            optimizer = _optimizers[optimizer_key](learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            grads_and_vars = [
                (None if g is None else tf.clip_by_value(
                    -gradient_clip_value, gradient_clip_value, g), v)
                for g, v in grads_and_vars]
            return optimizer.apply_gradients(grads_and_vars, global_step)

    return TrainModel(get_train_op, batch_size, max_steps)
