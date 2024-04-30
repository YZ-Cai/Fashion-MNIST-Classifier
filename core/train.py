import numpy as np
from core.utils import cross_entropy_loss


def train(images, labels, model):
    y = np.zeros((labels.size, 10))
    y[np.arange(labels.size), labels] = 1
    
    for i in range(100):
        y_pred = model.forward(images)
        loss = cross_entropy_loss(y, y_pred)
        pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(pred_labels == labels)
        print(f'Iter {i}. Loss: {loss}. Accuracy: {accuracy:.2f}')
        
        model.backward(y, y_pred)
        model.step(1e-5, 0)
        
        