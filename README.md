# 3D Deep Learning architectures to diagnose Alzheimer's Disease
Deep learning architectures to diagnose Alzheimer's Disease and Mild Cognitive Disorder using 3D neuroimages and patient metadata.

## Install Dependencies

*   Tensorflow 2.2.0
*   Numpy 1.18.4
*   matplotlib 3.2.1
*   sklearn 0.22.2
*   PyYAML 5.3.1

## Make TFRecords dataset
Reads image data from .npy and demographics data from .csv file and creates a TFRecord dataset file that is utilized to train models
```bash
npy_path=path/to/npy_files
records_path=path/to/output_dir
meta_path=path/to/demographics.csv
```
```bash
python make_tfrecords.py --numpy_data_path=$npy_path \
--out_path=$records_path \
--demographics_path=$meta_path
```

## Train Models
There are three types of models that are available in this project. Each having it's own config file at configs/. Models are trained by issuing the command below
```bash
python train.py --config_path=configs/conv_3d.yml \
--num_epochs=60 \
--early_stopping=True \
--save_model=True \
--save_weights=False
```

## Evaluate Models
Loads models from checkpoints and evaluates them

```bash
python evaluate.py --config_path=configs/conv_3d.yml
```

## Other related projects
Dataset obtained from Alzheimer's Disease Neuroimaging Initiative. Checkout these other related projects - https://github.com/jbrown81/ADNI_kaggle and https://github.com/regnerus/keras-alzheimers-3d-conv
