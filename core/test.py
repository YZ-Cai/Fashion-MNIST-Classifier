import numpy as np
from core.utils import Dataset
from core.model import Model



def test(images, labels, model: Model):
    y_pred = model.forward(images)
    pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_labels == labels)
    return accuracy