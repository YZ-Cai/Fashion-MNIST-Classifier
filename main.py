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
model_path = f'./checkpoints/hidden{hidden_sizes}_lr{init_lr}_gamma{gamma}_l2{l2_reg}.pkl'

# Load fashion MNIST dataset
train_dataset = Dataset(kind='train')
test_images, test_labels = Dataset(kind='t10k').get_all_data()

# initialize model
model = Model(train_dataset.x_dim, hidden_sizes, activation_type, train_dataset.y_dim)
model.init_params()

# train the model
train(train_dataset, model, model_path, init_lr=init_lr, gamma=gamma, l2_reg=l2_reg)

# test the model
model.load(model_path)
test_accuracy = test(test_images, test_labels, model)
print(f'\nTest accuracy: {test_accuracy*100:.1f}%')