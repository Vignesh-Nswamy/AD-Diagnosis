import tensorflow as tf
import pandas as pd
import numpy as np
import os


tf.compat.v1.flags.DEFINE_string('numpy_data_path', '', 'Path to training, val and test data contained in .npy files.')
tf.compat.v1.flags.DEFINE_string('out_path', '', 'Path where tfrecord files are stored')
tf.compat.v1.flags.DEFINE_string('demographics_path', '', 'Path to .csv file containing demographic information and labels')
FLAGS = tf.compat.v1.flags.FLAGS

labels = pd.read_csv(FLAGS.demographics_path)
write_options = tf.io.TFRecordOptions(compression_type='GZIP',
                                      compression_level=9)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_datasets(type: str):
    if type not in ['train', 'test', 'valid']: raise Exception('Unsupported dataset type')
    train_valid_test = 0 if type == 'train' else 1 if type == 'valid' else 2
    i = 1
    dataset = np.load(os.path.join(FLAGS.numpy_data_path, f'img_array_{type}_6k_{i}.npy'))
    while True:
        try:
            i += 1
            dataset = np.vstack((dataset, np.load(os.path.join(FLAGS.numpy_data_path, f'img_array_{type}_6k_{i}.npy'))))
        except FileNotFoundError:
            print(f'Loaded all {type} datasets')
            break
    for n in range(dataset.shape[0]):
        dataset[n, :, :] = dataset[n, :, :] / np.amax(dataset[n, :, :].flatten())
    print(f'Normalized {n+1} images')
    dataset = np.reshape(dataset, (-1, 62, 96, 96, 1))
    return dataset, labels[labels.train_valid_test == train_valid_test]


def create_tfrecords(file_name, img_data, demographics):
    assert img_data.shape[0] == demographics.shape[0]
    with tf.io.TFRecordWriter(os.path.join(FLAGS.out_path, file_name + '.tfrecords'), options=write_options) as writer:
        for i in np.random.choice(list(range(img_data.shape[0])), replace=False, size=(img_data.shape[0])):
            img_3d = img_data[i, :, :, :, :]
            dem_row = demographics.iloc[i]
            channels, height, width = img_3d.shape[0], img_3d.shape[1], img_3d.shape[2]
            img_raw = img_3d.tostring()
            onehot_label = np.eye(3)[dem_row.diagnosis - 1]
            label_raw = onehot_label.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_channels': _int64_feature(channels),
                'img_height': _int64_feature(height),
                'img_width': _int64_feature(width),
                'img_raw': _bytes_feature(img_raw),
                'sex': _bytes_feature(dem_row.sex.encode()),
                'age': _float_feature(dem_row.age_at_scan),
                'label': _bytes_feature(label_raw)
            }))
            writer.write(example.SerializeToString())
    writer.close()


def main(_):
    for data_type in ['train', 'valid', 'test']:
        data, demographics = load_datasets(data_type)
        create_tfrecords(data_type, data, demographics)


if __name__ == '__main__':
    tf.compat.v1.app.run()
