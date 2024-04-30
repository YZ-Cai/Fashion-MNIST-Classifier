from core.utils import Dataset
from core.model import Model
from core.train import train
from core.test import test



# configs
hidden_sizes = [512, 128]
activation_type = 'relu'

# Load MNIST data
train_dataset = Dataset(kind='train')
test_dataset = Dataset(kind='t10k')

# initialize model
model = Model(train_dataset.x_dim, hidden_sizes, activation_type, train_dataset.y_dim)

# train the model
train(train_dataset, model)

# test the model
test(test_dataset, model)