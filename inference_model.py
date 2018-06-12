from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_template import InferenceModel
from tf_template.visualization import ImageVis


class DepthDenoisingInferenceModel(InferenceModel):

    def get_inference(self, features, mode):
        raise NotImplementedError('TODO')

    def prediction_vis(self, prediction_data):
        return ImageVis(prediction_data)


class SpenDenoisingModel(DepthDenoisingInferenceModel):
    def __init__(self, layer_specs=None, optimizer_kwargs=None):
        if layer_specs is None:
            layer_specs = [
                {"filters": 32, "activation_spec": {"key": "softabs"}}
              ]
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                "key": "momentum",
                "n_iters": 20,
                "momentum": 0.75
            }
        self._layer_specs = layer_specs
        self._optimizer_kwargs = optimizer_kwargs

    def get_predictions(self, features, inference):
        return inference['opt']

    def get_initial_solution(self, features):
        return features

    def _get_energy_fn(self, features, mode):
        from .deserialize import deserialize_conv_spec

        with tf.variable_scope('energy'):
            sigma = tf.Variable(1.0, dtype=tf.float32, name='sigma')
            tf.summary.scalar('sigma2', sigma)
            layers = [
                deserialize_conv_spec(**spec) for spec in self._layer_specs]
            input_shape = tf.TensorShape(features.shape)
            for layer in layers:
                layer.build(input_shape)
                input_shape = layer.compute_output_shape(input_shape)

        def f(y):
            with tf.name_scope('energy_eval'):
                local_energy = tf.reduce_sum(tf.square(features - y))
                p = y
                for layer in layers:
                    p = layer(p)
                global_energy = tf.reduce_sum(p)

                total_energy = local_energy + 2*sigma*global_energy
            return total_energy

        return f

    def _get_inner_optimizer(self):
        from .opt import get_inner_optimizer
        return get_inner_optimizer(**self._optimizer_kwargs)

    def get_inference(self, features, mode):
        energy_fn = self._get_energy_fn(features, mode)
        inner_optimizer = self._get_inner_optimizer()
        x0 = self.get_initial_solution(features)
        ys = inner_optimizer.minimize(
            energy_fn, x0, back_prop=True)
        opt = ys[-1]
        return dict(opt=opt, steps=ys)


def get_inference_model(implementation='spen', **kwargs):
    implementation = implementation.lower()
    if implementation == 'spen':
        return SpenDenoisingModel(**kwargs)
    else:
        raise NotImplementedError(
            'Unrecognized implementation "%s"' % implementation)
