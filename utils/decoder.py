import tensorflow as tf


def single_input_decoder(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """

    features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'img_channels': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_height': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_width': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_raw': tf.compat.v1.FixedLenFeature([], tf.string),
            'sex': tf.compat.v1.FixedLenFeature([], tf.string),
            'age': tf.compat.v1.FixedLenFeature([], tf.float32),
            'label': tf.compat.v1.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    channels = tf.cast(features['img_channels'], tf.int32)
    image = tf.compat.v1.decode_raw(features['img_raw'], tf.float64)
    image = tf.reshape(image, (channels, height, width, 1))
    label = tf.compat.v1.decode_raw(features['label'], tf.float64)
    return image, label


def mixed_input_decoder(serialized_example):
    """
    Parses an image, a numerical vector and a label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """

    features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'img_channels': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_height': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_width': tf.compat.v1.FixedLenFeature([], tf.int64),
            'img_raw': tf.compat.v1.FixedLenFeature([], tf.string),
            'sex': tf.compat.v1.FixedLenFeature([], tf.string),
            'age': tf.compat.v1.FixedLenFeature([], tf.float32),
            'label': tf.compat.v1.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    channels = tf.cast(features['img_channels'], tf.int32)
    image = tf.compat.v1.decode_raw(features['img_raw'], tf.float64)
    image = tf.reshape(image, (channels, height, width, 1))
    label = tf.compat.v1.decode_raw(features['label'], tf.float64)
    sex = tf.cast(1 if features['sex'] == 'M' else 2, tf.float64)
    age = tf.cast(features['age'] / 100, tf.float64)
    return {'image_input': image, 'num_input': tf.stack([sex, age])}, label