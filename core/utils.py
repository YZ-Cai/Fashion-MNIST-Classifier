import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    x_dim = 784
    y_dim = 10
    
    def __init__(self, data_path='./data', kind="train"):
        """
        Load MNIST data from `path`
        Codes from: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
        """
        labels_path = os.path.join(data_path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(data_path, '%s-images-idx3-ubyte.gz' % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            self._labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            self._images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self._labels), self.x_dim) / 255.0
    
    
    def get_all_data(self):
        """
        Get all data
        """
        return self._images, self._labels
    
    
    def split_validation_data(self, train_ratio=0.8):
        """
        Split data into training and validation sets, return validation data
        """
        
        # shuffle data
        idx = np.random.permutation(len(self._labels))
        self._images = self._images[idx]
        self._labels = self._labels[idx]
        
        # split data
        self._num_train = int(len(self._labels) * train_ratio)
        self._train_images = self._images[:self._num_train]
        self._train_labels = self._labels[:self._num_train]
        return self._images[self._num_train:], self._labels[self._num_train:]
    
    
    def init_batch(self, batch_size=128):
        """
        Initialize batch data, return number of batches
        """
        
        # shuffle training data
        idx = np.random.permutation(self._num_train)
        self._train_images = self._train_images[idx]
        self._train_labels = self._train_labels[idx]
        
        # init for first batch
        self._batch_size = batch_size
        return self._num_train // batch_size
    
    
    def get_batch_data(self, idx):
        """
        Get batch data given batch idx
        """
        start = idx * self._batch_size
        end = start + self._batch_size
        return self._train_images[start:end], self._train_labels[start:end]



def cross_entropy_loss(labels, y_pred):
    """
    Cross entropy loss
    """
    y = np.zeros((labels.size, 10))
    y[np.arange(labels.size), labels] = 1
    return -np.sum(y * np.log(y_pred+1e-7)) / labels.size



def plot_curves(file_prefix):
    """
    Plot loss and accuracy curves
    """
    df_loss_accuracy = pd.read_csv(file_prefix+'.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # loss curve
    ax1.plot(df_loss_accuracy['epoch'], df_loss_accuracy['train_loss'], label='Train Loss')
    ax1.plot(df_loss_accuracy['epoch'], df_loss_accuracy['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # accuracy curve
    ax2.plot(df_loss_accuracy['epoch'], df_loss_accuracy['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(file_prefix+'.png')