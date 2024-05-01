import numpy as np
from core.utils import Dataset
from core.model import Model
from core.train import train
from core.test import test



# set random seeds
np.random.seed(2024)

# configs
hidden_sizes = [512, 128]
activation_type = 'relu'

# Load fashion MNIST dataset
train_dataset = Dataset(kind='train')
test_images, test_labels = Dataset(kind='t10k').get_all_data()

# initialize model
model = Model(train_dataset.x_dim, hidden_sizes, activation_type, train_dataset.y_dim)

# train the model
train(train_dataset, model)

# test the model
test_accuracy = test(test_images, test_labels, model)
print(f'Test accuracy: {test_accuracy*100:.1f}%')