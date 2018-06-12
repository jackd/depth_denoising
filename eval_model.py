from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template.eval_model import EvalModel
import numpy as np


def get_loss_fn(use_intermediate_loss, step_loss_fn):
    if use_intermediate_loss:
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


def get_eval_model(use_intermediate_losses=True):
    def step_loss(step_inference, labels):
        return tf.reduce_mean((step_inference - labels)**2)
    inference_loss = get_loss_fn(use_intermediate_losses, step_loss)

    def get_eval_metric_ops(predictions, labels):
        from .ops import get_psnr
        predictions = tf.clip_by_value(predictions, 0, 1)
        psnr = get_psnr(predictions, labels, 1.0)
        psnr = tf.metrics.mean(psnr)
        return dict(psnr=psnr)

    return EvalModel(inference_loss, get_eval_metric_ops)
