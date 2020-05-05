from .base import BaseModel

import tensorflow as tf
from tensorflow.keras.models import Model as keras_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class ConvThreeD(BaseModel):
    def __init__(self,
                 config):
        super(ConvThreeD, self).__init__(config)

    def create(self):
        input_layer = Input(shape=eval(self.config.input.shape),
                            name='input_layer')
        # First conv layer
        conv_1 = Conv3D(filters=8,
                        kernel_size=3,
                        dilation_rate=1,
                        strides=2,
                        activation=LeakyReLU(0.2),
                        kernel_initializer=self._he_normal_initializer,
                        kernel_regularizer=self._kernel_regularizer,
                        name='conv_1')(input_layer)
        batch_norm_1 = batch_norm_1 = BatchNormalization(name='batch_norm_1')(conv_1)
        # Second conv layer
        conv_2 = Conv3D(filters=8,
                        kernel_size=3,
                        dilation_rate=1,
                        strides=2,
                        activation=LeakyReLU(0.2),
                        kernel_initializer=self._he_normal_initializer,
                        kernel_regularizer=self._kernel_regularizer,
                        name='conv_2')(batch_norm_1)
        batch_norm_2 = BatchNormalization(name='batch_norm_2')(conv_2)
        # Third conv layer
        conv_3 = Conv3D(filters=8,
                        kernel_size=3,
                        dilation_rate=1,
                        strides=2,
                        activation=LeakyReLU(0.2),
                        kernel_initializer=self._he_normal_initializer,
                        kernel_regularizer=self._kernel_regularizer,
                        name='conv_3')(batch_norm_2)
        batch_norm_3 = BatchNormalization(name='batch_norm_3')(conv_3)
        # Flatten
        flatten = Flatten(name='flatten')(batch_norm_3)
        # Dense layer 1
        dense_1 = Dense(1024,
                        activation=LeakyReLU(),
                        kernel_initializer=self._he_normal_initializer,
                        kernel_regularizer=self._kernel_regularizer,
                        name='dense_1')(flatten)
        drop_1 = Dropout(0.5,
                         name='dropout_1')(dense_1)
        # Dense layer 2
        dense_2 = Dense(512,
                        activation=LeakyReLU(),
                        kernel_initializer=self._he_normal_initializer,
                        kernel_regularizer=self._kernel_regularizer,
                        name='dense_2')(drop_1)
        drop_2 = Dropout(0.5,
                         name='dropout_2')(dense_2)
        # Output layer
        output_layer = Dense(3,
                             activation='softmax',
                             kernel_initializer=self._lecun_normal_initializer,
                             name='output')(drop_2)
        model = keras_model(input_layer, output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=self._optimizer,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

        return model
