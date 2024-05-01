import numpy as np
import pandas as pd
from tqdm import tqdm
from core.utils import Dataset, cross_entropy_loss
from core.model import Model
from core.test import test



def train(dataset: Dataset, model: Model, file_prefix, train_ratio=0.95, batch_size=1024, num_epoch=20, init_lr=2e-3, gamma=0.95, l2_reg=1e-4):
    
    # validation for saving the best model
    val_images, val_labels = dataset.split_validation_data(train_ratio)
    best_val_accuracy = 0
    df_loss_accuracy = pd.DataFrame()

    # iterate each epoch
    for epoch in range(num_epoch):
        num_batch = dataset.init_batch(batch_size)
        train_losses, train_accuracies = [], []
        
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
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            progress_bar.set_postfix({'Train loss': loss, 'Train accuracy': accuracy})
            progress_bar.update(1)
        
        # validation
        val_loss, val_accuracy = test(val_images, val_labels, model)
        train_loss = np.mean(np.array(train_losses))
        train_accuracy = np.mean(np.array(train_accuracies))
        
        # print and record loss and accuracy
        progress_bar.set_postfix({'Train loss': train_loss, 'Train acc': train_accuracy, 'Val loss': val_loss, 'Val acc': val_accuracy})
        progress_bar.update(1)
        df_loss_accuracy = pd.concat([df_loss_accuracy, pd.DataFrame([{
            'epoch': epoch+1, 
            'train_loss': train_loss, 'train_accuracy': train_accuracy, 
            'val_loss': val_loss, 'val_accuracy': val_accuracy
            }])], ignore_index=True)
        
        # save model if validation accuracy is improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save(file_prefix+'.pkl')

    # save loss and accuracy
    df_loss_accuracy.to_csv(file_prefix+'.csv', index=False)