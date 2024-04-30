import numpy as np



def test(images, labels, model):
    y_pred = model.forward(images)
    pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_labels == labels)
    print(f'Accuracy: {accuracy:.2f}')