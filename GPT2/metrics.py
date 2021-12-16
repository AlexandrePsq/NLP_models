import numpy as np
from sklearn import metrics

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
            return metrics.classification_report(y_true, y_pred)
        elif metric_name=='explained_variance':
            return metrics.explained_variance_score(y_true, y_pred)
        elif metric_name=='max_error':
            return metrics.max_error(y_true, y_pred)
        elif metric_name=='neg_mean_absolute_error':
            return metrics.mean_absolute_error(y_true, y_pred)
        elif metric_name=='neg_mean_squared_error':
            return metrics.mean_squared_error(y_true, y_pred)
        elif metric_name=='neg_root_mean_squared_error':
            return metrics.mean_squared_error(y_true, y_pred)
        elif metric_name=='neg_mean_squared_log_error':
            return metrics.mean_squared_log_error(y_true, y_pred)
        elif metric_name=='neg_median_absolute_error':
            return metrics.median_absolute_error(y_true, y_pred)
        elif metric_name=='r2':
            return metrics.r2_score(y_true, y_pred)