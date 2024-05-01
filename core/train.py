import numpy as np
from tqdm import tqdm
from core.utils import Dataset, cross_entropy_loss
from core.model import Model
from core.test import test



def train(dataset: Dataset, model: Model, model_path, train_ratio=0.95, batch_size=1024, num_epoch=20, init_lr=2e-3, gamma=0.95, l2_reg=1e-4):
    
    # validation for saving the best model
    val_images, val_labels = dataset.split_validation_data(train_ratio)
    best_val_accuracy = 0

    # iterate each epoch
    for epoch in range(num_epoch):
        num_batch = dataset.init_batch(batch_size)
        losses, accuracies = [], []
        
        # learning rate decay
        lr = init_lr * (gamma**epoch)
        
        # iterate each batch
        progress_bar = tqdm(total=num_batch+1, desc=f'Epoch {epoch+1}')
        for idx in range(num_batch):
            images, labels = dataset.get_batch_data(idx)
            
            # forward pass
            y_pred = model.forward(images)
            
            # loss and accuracy
            loss = cross_entropy_loss(labels, y_pred)
            pred_labels = np.argmax(y_pred, axis=1)
            accuracy = np.mean(pred_labels == labels)
            
            # backward pass and update weights
            model.backward(labels, y_pred)
            model.step(lr, l2_reg)
            
            # record loss and accuracy
            losses.append(loss)
            accuracies.append(accuracy)
            progress_bar.set_postfix({'Train loss': loss, 'Train accuracy': accuracy})
            progress_bar.update(1)
        
        # validation
        val_accuracy = test(val_images, val_labels, model)
        progress_bar.set_postfix({'Train loss': np.mean(np.array(losses)), 'Train accuracy': np.mean(np.array(accuracies)), 'Val accuracy': val_accuracy})
        progress_bar.update(1)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save(model_path)
