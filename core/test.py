import numpy as np
from core.utils import Dataset
from core.model import Model



def test(dataset: Dataset, model: Model):
    
    # get all test data
    images, labels = dataset.get_all_data()
    
    # forward pass
    y_pred = model.forward(images)
    pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_labels == labels)
    print(f'Test accuracy: {accuracy:.2f}')