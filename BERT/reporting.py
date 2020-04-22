"""
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Report(object):
    """Class regrouping functions to display results.
    """

    @classmethod
    def plots_train_val_loss(cls, training_stats, nb_epochs, output_path, test_accuracy=None, test_loss=None):
        """ Plot train and validation losses over fine-tuning.
        """
        df = pd.DataFrame(data=training_stats)
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df['Training Loss'], 'b-o', label="Training")
        plt.plot(df['Valid. Loss'], 'g-o', label="Validation")
        if test_loss:
            plt.hlines(test_loss, 0, nb_epochs, 'r', label='Test')

        # Label the plot.
        if test_loss:
            plt.title("Training, Validation & Test Loss")
        else:
            plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(1, nb_epochs + 1))
        plt.savefig(output_path)
        plt.show()