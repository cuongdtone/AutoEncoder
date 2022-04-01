import warnings
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
def plot_cm(CM, normalize=True, save_dir='', names=(), show=True):
    if True:
        import seaborn as sn
        array = CM/ ((CM.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if 2 < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == 2  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=2 < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names if labels else "auto",
                       yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        if show:
            plt.show()
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close()