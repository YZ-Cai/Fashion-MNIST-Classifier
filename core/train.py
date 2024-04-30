import numpy as np
from core.utils import cross_entropy_loss, Dataset
from core.model import Model



def train(dataset: Dataset, model: Model, train_ratio=0.8, batch_size=1024, init_lr=1e-5, l2_reg=0.0):
    
    # split training and validation data
    val_images, val_labels = dataset.split_validation_data(train_ratio)
    
    # iterate each epoch
    for epoch in range(100):
        dataset.init_batch(batch_size)
        
        # iterate each batch
        while True:
            images, labels = dataset.get_batch_data()
            if images is None:
                break
            
            # forward pass
            y_pred = model.forward(images)
            
            # loss and accuracy
            loss = cross_entropy_loss(labels, y_pred)
            pred_labels = np.argmax(y_pred, axis=1)
            accuracy = np.mean(pred_labels == labels)
            
            # backward pass and update weights
            model.backward(labels, y_pred)
            model.step(init_lr, l2_reg)
            print(f'Epoch {epoch+1}. Train loss: {loss:.2f}, train accuracy: {accuracy*100:.1f}%', end='\r')
        
        # validation
        print(f'Epoch {epoch+1}. Train loss: {loss:.2f}, train accuracy: {accuracy*100:.1f}%')
