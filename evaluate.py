import os
import yaml
import tensorflow as tf
from utils.model_utils import evaluate
from utils.model_utils import load_checkpoint
from utils import decoder
from utils import util

tf.compat.v1.flags.DEFINE_string('config_path', '', 'Path to a YAML configuration files defining FLAG values.')
FLAGS = tf.compat.v1.flags.FLAGS


def main(_):
    config = yaml.load(open(FLAGS.config_path), Loader=yaml.FullLoader)
    config = util.merge(config,
                        FLAGS)

    data_decoder = decoder.mixed_input_decoder if config.input.type == 'mixed' \
        else decoder.single_input_decoder
    test_dataset = tf.data.TFRecordDataset(config.data_paths.test,
                                           compression_type='GZIP').map(data_decoder).batch(1)
    model_stats = evaluate(load_checkpoint(os.path.join(config.checkpoint_dir, 'checkpoint.ckpt')),
                           test_dataset,
                           config.input.type)


if __name__ == '__main__':
    tf.compat.v1.app.run()
