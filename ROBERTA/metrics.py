import numpy as np
from sklearn.metrics import classification_report


class Metrics(object):
    """Define metrics to be use when training/evaluating models.
    """

    @classmethod
    def flat_accuracy(cls, y_true, y_pred):
        """ Function to calculate the accuracy of our predictions vs labels
        """
        pred_flat = np.argmax(y_pred, axis=-1).flatten()
        labels_flat = y_true.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @classmethod
    def report(cls, metric_name, y_true, y_pred):
        if metric_name=='classification':
            return classification_report(y_true, y_pred)