import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#%tensorflow_version 2.x
import tensorflow as tf

class stats:
    #Do not change this function header
    def __init__(self,model):
        self.model = model   
        def cf(self,x=[]):
            x = np.asarray(x)
            y =[]
            for i in range(len(x)):
                max = np.argmax(x[i])
                y.append(max)
            return np.asarray(y)

        def load_test_dataset(self):
            i = 1
            data_path = 'numpy_data'
            dataset = np.load(os.path.join(data_path, f'img_array_test_6k_{i}.npy'))
            while True:
                try:
                    i += 1
                    dataset = np.vstack((dataset, np.load(os.path.join(data_path, f'img_array_test_6k_{i}.npy'))))
                except FileNotFoundError:
                    print(f'Loaded all test datasets')
                    break
            # dataset = np.expand_dims(dataset, axis=1)
            for n in range(dataset.shape[0]):
                dataset[n, :, :] = dataset[n, :, :] / np.amax(dataset[n, :, :].flatten())
            print(f'Normalized {n+1} images')
            dataset = np.reshape(dataset, (-1, 62, 96, 96, 1))
            return dataset, np.eye(3)[labels[labels.train_valid_test == 2]['diagnosis'] - 1]

        labels = pd.read_csv('/content/drive/Shared drives/IML Project/Project/demographics.csv')
        test_data, test_labels = self.load_test_dataset()

        # Predicted Test labels
        predictions = self.model.predict(test_data)

        # Modified actual Test labels
        y_test = self.cf(test_labels)


        # Modified predicted test labels
        y_pred = self.cf(predictions)

        # Confusion Matrix
        conf = metrics.confusion_matrix(y_test, y_pred)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred, average = None)
        f1_score_avg  = metrics.f1_score(y_test, y_pred, average = 'macro')
        recall = metrics.recall_score(y_test, y_pred, average= None)

        return accuracy, f1_score, f1_score_avg, recall



