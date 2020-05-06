import os

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

from utils import decoder
from utils.model_utils import load_model
from utils.model_utils import evaluate


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self,
                 config):
        self.config = config

        # Save directories
        self._checkpoint_dir = config.checkpoint_dir
        self._weights_dir = config.weights_dir

        # Learning parameters
        self._lr = eval(config.learning.rate)
        self._decay_factor = config.learning.decay
        self._decay_patience = config.learning.decay_patience
        self._optimizer = RMSprop(self._lr) if config.training.optimizer == 'rmsprop' \
            else SGD(self._lr) if config.training.optimizer == 'sgd' \
            else Adam(self._lr)

        # Default params
        self._kernel_regularizer = tf.keras.regularizers.l2(0.001)
        self._he_normal_initializer = tf.keras.initializers.he_normal(seed=0)
        self._lecun_normal_initializer = tf.keras.initializers.lecun_normal(seed=0)

        # Pretraining operations
        self.__create_dirs()
        self.callbacks = self.__get_callbacks()

        # Create model
        self.__init_model()

        #data decoder
        self.data_decoder = decoder.mixed_input_decoder if self.config.input.type == 'mixed' \
            else decoder.single_input_decoder

    def __create_dirs(self):
        if not os.path.exists(self._checkpoint_dir):
            print(f'{self._checkpoint_dir} not found. Creating....')
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        if not os.path.exists(self._weights_dir):
            print(f'{self._weights_dir} not found. Creating....')
            os.makedirs(self._weights_dir, exist_ok=True)

    def __get_callbacks(self):
        callback_list = list()
        callback_list.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                                                  factor=self._decay_factor,
                                                                  patience=self._decay_patience,
                                                                  min_delta=0.01,
                                                                  verbose=1))
        if self.config.save_model:
            callback_list.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self._checkpoint_dir, 'checkpoint.ckpt'),
                                                                    monitor='val_categorical_accuracy',
                                                                    verbose=1,
                                                                    save_best_only=True,
                                                                    mode='max'))
        if self.config.save_weights:
            callback_list.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self._weights_dir, 'best_weights.h5'),
                                                                    monitor='val_categorical_accuracy',
                                                                    verbose=1,
                                                                    save_best_only=True,
                                                                    save_weights_only=True,
                                                                    mode='max'))
        if self.config.early_stopping:
            callback_list.append(tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                                  min_delta=1e-3,
                                                                  patience=7,
                                                                  verbose=1,
                                                                  mode='max'))
        return callback_list

    def __init_model(self):
        if os.path.exists(os.path.join(self._checkpoint_dir, 'checkpoint.ckpt')):
            print(f'Found model at {os.path.join(self._checkpoint_dir, "checkpoint.ckpt")}. Loading...')
            self.model = load_model(os.path.join(self._checkpoint_dir, 'checkpoint.ckpt'))
        else:
            print(f'No model found at {os.path.join(self._checkpoint_dir, "checkpoint.ckpt")}. Creating...')
            self.model = self.create()
        print(self.model.summary())

    @abstractmethod
    def create(self):
        pass

    def train(self):
        train_dataset = tf.data.TFRecordDataset(self.config.data_paths.train,
                                                compression_type='GZIP').map(self.data_decoder).batch(self.config.training.batch_size)
        val_dataset = tf.data.TFRecordDataset(self.config.data_paths.validation,
                                              compression_type='GZIP').map(self.data_decoder).batch(self.config.training.batch_size)
        self.model.fit(train_dataset,
                       validation_data=val_dataset,
                       epochs=self.config.num_epochs,
                       callbacks=self.callbacks,
                       verbose=1)

    def evaluate(self):
        test_dataset = tf.data.TFRecordDataset(self.config.data_paths.test,
                                               compression_type='GZIP').map(self.data_decoder).batch(1)
        return evaluate(self.model, test_dataset, self.config.input.type)
