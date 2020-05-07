from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import luatables


def get_stats(y_true, y_pred):
    stats_dict = luatables.LuaEmulator()
    stats_dict.accuracy = accuracy_score(y_true, y_pred)
    stats_dict.confusion_matrix = confusion_matrix(y_true, y_pred)
    stats_dict.f1_scores = f1_score(y_true, y_pred, average=None)
    stats_dict.average_f1 = f1_score(y_true, y_pred, average='macro')
    stats_dict.recall_scores = recall_score(y_true, y_pred, average=None)
    stats_dict.average_recall = recall_score(y_true, y_pred, average='macro')
    display_stats(stats_dict)


def display_stats(stats):
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(f'Classification accuracy: {stats.accuracy}')
    print(f'F1 Score: {stats.average_f1}')
    print(f'Individual f1 scores: {stats.f1_scores}')
    print(f'Recall: {stats.average_recall}')
    print(f'Individual recalls: {stats.recall_scores}')
    df_cm = pd.DataFrame(stats.confusion_matrix.astype('int32'), range(3), range(3))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

