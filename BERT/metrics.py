import numpy as np
from sklearn.metrics import classification_report


class Metrics(object):
    """Define metrics to be use when training/evaluating models.
    """

    @classmethod
    def flat_accuracy(cls, y_true, y_pred):
        """ Function to calculate the accuracy of our predictions vs labels
        """
        return np.sum(y_pred == y_true) / len(y_true)

    @classmethod
    def report(cls, metric_name, y_true, y_pred):
        if metric_name=='classification':
            return classification_report(y_true, y_pred)