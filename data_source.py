from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template import DataSource
from tf_template.visualization import ImageVis

from seven_scenes.preprocessed import index_filenames
from seven_scenes.preprocessed import load_bin_data, image_shape, max_gt_value


def get_seven_scenes_inputs(
        mode, batch_size, n_epochs=None, crop_size=(96, 128), max_val=5000):
    if max_val is None:
        max_val = max_gt_value
    index = index_filenames()
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if is_training:
        index = {'chess': index['chess']}
    else:
        del index['chess']
    all_raw = []
    all_ground_truth = []
    for v in index.values():
        for raw, ground_truth in v.values():
            all_raw.append(raw)
            all_ground_truth.append(ground_truth)
    raw = tf.convert_to_tensor(all_raw, dtype=tf.string)
    ground_truth = tf.convert_to_tensor(all_ground_truth, dtype=tf.string)

    def py_map_fn(raw_fn, gt_fn):
        raw = load_bin_data(raw_fn)
        gt = load_bin_data(gt_fn)
        # raw[raw == 0] = max_val
        # gt[gt == 0] = max_val
        return raw, gt

    def tf_map_fn(raw_fn, gt_fn):
        raw, gt = tf.py_func(
            py_map_fn, (raw_fn, gt_fn), (tf.int16, tf.int16),
            stateful=False)
        if is_training:
            # crop
            max_crops = [s - c for (s, c) in zip(image_shape, crop_size)]
            cy, cx = (
                tf.random_uniform(shape=(), maxval=m, dtype=tf.int32)
                for m in max_crops)
            h, w = crop_size
            raw = raw[cy:cy + h, cx:cx + w]
            gt = gt[cy:cy + h, cx:cx + w]
            shape = crop_size
        else:
            shape = image_shape

        def preprocess(x):
            x.set_shape(shape)
            return tf.minimum(tf.cast(tf.expand_dims(
                x, axis=-1), tf.float32) / max_val, 1)

        raw = preprocess(raw)
        gt = preprocess(gt)
        return raw, gt

    dataset = tf.data.Dataset.from_tensor_slices((raw, ground_truth))
    if is_training:
        dataset = dataset.repeat(n_epochs)
    dataset = dataset.shuffle(len(all_raw))
    dataset = dataset.map(tf_map_fn, num_parallel_calls=8)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
    raw, gt = dataset.make_one_shot_iterator().get_next()
    return raw, gt


class DepthDenoisingDataSource(DataSource):
    def __init__(self, dataset_id='seven_scenes'):
        self._dataset_id = dataset_id

    @property
    def dataset_id(self):
        return self._dataset_id

    def get_inputs(self, mode, batch_size=None):
        return get_seven_scenes_inputs(mode, batch_size)

    def feature_vis(self, features):
        return ImageVis(features)

    def label_vis(self, label):
        return ImageVis(label)


def get_data_source(dataset_id='seven_scenes'):
    return DepthDenoisingDataSource(dataset_id=dataset_id)
