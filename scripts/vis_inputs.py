#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
from tf_template.cli import vis_inputs
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_id', default='seven_scenes', help='id of depth dataset')


def main(_):
    from depth_denoising.data_source import get_data_source
    data_source = get_data_source(FLAGS.dataset_id)
    vis_inputs(data_source)


app.run(main)
