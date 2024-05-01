import numpy as np
from core.utils import Dataset
from core.model import Model
from core.train import train
from core.test import test



# set random seeds
np.random.seed(0)

# configs
hidden_sizes = [512, 128]
activation_type = 'relu'
init_lr=2e-3
gamma=0.95
l2_reg=1e-4
train_ratio = 0.95

# Load fashion MNIST dataset
train_dataset = Dataset(kind='train')
val_images, val_labels = train_dataset.split_validation_data(train_ratio)
test_images, test_labels = Dataset(kind='t10k').get_all_data()

# initialize model
model = Model(train_dataset.x_dim, hidden_sizes, train_dataset.y_dim, activation_type)
model.init_params()

# train the model
file_prefix = f'./outputs/hidden{hidden_sizes}_lr{init_lr}_gamma{gamma}_l2{l2_reg}'
train(train_dataset, model, val_images, val_labels, file_prefix, init_lr=init_lr, gamma=gamma, l2_reg=l2_reg)

# test the model
model.load(file_prefix+'.pkl')
test_loss, test_accuracy = test(test_images, test_labels, model)
print(f'\nTest accuracy: {test_accuracy*100:.1f}%')