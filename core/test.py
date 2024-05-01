import numpy as np
from core.utils import cross_entropy_loss
from core.model import Model



def test(images, labels, model: Model):
    y_pred = model.forward(images)
    pred_labels = np.argmax(y_pred, axis=1)
    loss = cross_entropy_loss(labels, y_pred)
    accuracy = np.mean(pred_labels == labels)
    return loss, accuracy