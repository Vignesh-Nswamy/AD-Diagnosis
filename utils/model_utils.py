from tensorflow.keras.models import load_model
import numpy as np
from utils.eval_stats import get_stats


def load_checkpoint(ckpt_path):
    return load_model(ckpt_path)


def evaluate(model, data, model_type):
    predictions = list()
    true_classes = list()
    for input_data, label in data:
        inp = [input_data['image_input'].numpy(), input_data['num_input'].numpy()] if model_type == 'mixed' \
            else input_data.numpy()
        predicted_label = np.argmax(model.predict(inp))
        predictions.append(predicted_label)
        true_label = np.argmax(label.numpy())
        true_classes.append(true_label)

    return get_stats(true_classes, predictions)
