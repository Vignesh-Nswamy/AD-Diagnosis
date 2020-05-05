from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from utils import luatables


def get_stats(y_true, y_pred):
    stats_dict = luatables.LuaEmulator()
    stats_dict.accuracy = accuracy_score(y_true, y_pred)
    stats_dict.confusion_matrix = confusion_matrix(y_true, y_pred)
    stats_dict.f1_scores = f1_score(y_true, y_pred, average=None)
    stats_dict.average_f1 = f1_score(y_true, y_pred, average='macro')
    stats_dict.recall_scores = recall_score(y_true, y_pred, average=None)
    stats_dict.average_recalls = recall_score(y_true, y_pred, average='macro')
    return stats_dict

