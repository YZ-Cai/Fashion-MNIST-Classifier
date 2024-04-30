from core.utils import load_mnist
from core.model import Model
from core.train import train
from core.test import test


data_path = './data'
images, labels = load_mnist(data_path, kind='train')
model = Model(images.shape[1], [512, 128], 'relu', 10)
train(images, labels, model)

images, labels = load_mnist(data_path, kind='t10k')
test(images, labels, model)