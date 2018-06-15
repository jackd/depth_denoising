from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel
import tensorflow as tf
import numpy as np


def get_loss_fn(use_intermediate_losses, step_loss_fn):
    if use_intermediate_losses:
        def inference_loss(inference, labels):
            steps = inference['steps']
            T = len(steps)
            losses = []
            weights = []
            for t, step in enumerate(steps):
                weight = 1 / (T - t)
                loss = step_loss_fn(step, labels)
                tf.summary.scalar('loss%d' % t, loss, family='step_loss')
                weighted_loss = weight * loss
                losses.append(weighted_loss)
                weights.append(weight)
            tf.summary.scalar('opt_loss', loss)
            loss = tf.add_n(losses) / np.sum(weights)
            return loss
        return inference_loss
    else:
        def inference_loss(inference, labels):
            loss = step_loss_fn(inference['opt'], labels)
            tf.summary.scalar('opt_loss', loss)
            return loss
        return inference_loss


_optimizers = {
    'adam': tf.train.AdamOptimizer,
}


def get_train_model(
        batch_size=64, max_steps=10000,
        use_intermediate_losses=True,
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

    loss_fn = get_loss_fn(
        use_intermediate_losses, lambda x, y: tf.nn.l2_loss(x - y))

    return TrainModel.from_fns(loss_fn, get_train_op, batch_size, max_steps)
