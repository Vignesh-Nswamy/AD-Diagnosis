import yaml
import tensorflow as tf
from models import ConvThreeD
from models import Parallel
from models import MixedInput
from utils import util

tf.compat.v1.flags.DEFINE_string('config_path', '', 'Path to a YAML configuration files defining FLAG values.')
tf.compat.v1.flags.DEFINE_integer('num_epochs', 75, 'Number of training epochs')
tf.compat.v1.flags.DEFINE_integer('evaluate', True, 'Whether to evaluate model after training')
tf.compat.v1.flags.DEFINE_bool('early_stopping', True, 'Stop training early')
tf.compat.v1.flags.DEFINE_bool('save_model', True, 'Whether to save model checkpoint when val accuracy increases')
tf.compat.v1.flags.DEFINE_bool('save_weights', True, 'Whether to save model weights when val accuracy increases')
FLAGS = tf.compat.v1.flags.FLAGS


def main(_):
    config = yaml.load(open(FLAGS.config_path), Loader=yaml.FullLoader)
    config = util.merge(config,
                        FLAGS)
    classifier = MixedInput(config) if config.model_arch == 'mixed' \
        else Parallel(config) if config.model_arch == 'parallel' \
        else ConvThreeD(config)

    classifier.train()
    if config.evaluate:
        classifier.evaluate()


if __name__ == '__main__':
    tf.app.run()
