#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags, app
from tf_template.cli import coord_main

flags.DEFINE_string('model_id', default='spen_foe20', help='id of model')


def main(_):
    from depth_denoising.coordinator import get_coordinator
    coord_main(get_coordinator(flags.FLAGS.model_id))


app.run(main)
