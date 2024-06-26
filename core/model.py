import pickle
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, input_size, hidden_sizes, output_size, activation_type='relu'):
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_hidden_layers = len(hidden_sizes)
        self._output_size = output_size
        self._activation_type = activation_type
        
        
    def init_params(self):
        """
        Initialize weights and biases for the network
        """
        self._weights = []
        self._biases = []
        for i in range(self._num_hidden_layers):
            if i == 0:
                self._weights.append(np.random.randn(self._input_size, self._hidden_sizes[i]))
            else:
                self._weights.append(np.random.randn(self._hidden_sizes[i-1], self._hidden_sizes[i]))
            self._biases.append(np.random.randn(self._hidden_sizes[i]))
        self._weights.append(np.random.randn(self._hidden_sizes[-1], self._output_size))
        self._biases.append(np.random.randn(self._output_size))
            
    
    def load(self, model_file):
        self._weights, self._biases = pickle.load(open(model_file, 'rb'))
    
    
    def save(self, model_file):
        pickle.dump((self._weights, self._biases), open(model_file, 'wb'))


    def _activate(self, input):
        """
        Activation function
        """
        if self._activation_type == 'relu':
            return np.maximum(input, 0)
        elif self._activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-input))
        elif self._activation_type == 'tanh':
            return np.tanh(input)
        else:
            raise ValueError(f'Invalid activation type: {self._activation_type}')
        
        
    def _dactivate(self, output):
        """
        Derivative of activation function
        """
        if self._activation_type == 'relu':
            return (output > 0).astype(float)
        elif self._activation_type == 'sigmoid':
            return output * (1 - output)
        elif self._activation_type == 'tanh':
            return 1 - output**2
        else:
            raise ValueError(f'Invalid activation type: {self._activation_type}')
        
        
    def _softmax(self, input):
        """
        Softmax function
        """
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
        
        
    def forward(self, X):
        """
        Forward pass through the network
        """
        self._as = [X]
        for i in range(self._num_hidden_layers):
            self._as.append(self._activate(self._as[-1] @ self._weights[i] + self._biases[i]))
        y_pred = self._softmax(self._as[-1] @ self._weights[-1] + self._biases[-1])
        return y_pred
        
        
    def backward(self, labels, y_pred):
        """
        Backward pass through the network
        """
        
        # init gradients
        self._dzs = [None] * (self._num_hidden_layers + 1)
        self._dweights = [None] * (self._num_hidden_layers + 1)
        self._dbiases = [None] * (self._num_hidden_layers + 1)
        
        # one-hot encoding of labels
        y = np.zeros((labels.size, 10))
        y[np.arange(labels.size), labels] = 1
        batch_size = y.shape[0]
        
        # calculate gradients
        self._dzs[-1] = y_pred - y
        self._dweights[-1] = self._as[-1].T @ self._dzs[-1] / batch_size
        self._dbiases[-1] = np.mean(self._dzs[-1], axis=0)
        for i in range(self._num_hidden_layers, 0, -1):
            self._dzs[i-1] = self._dzs[i] @ self._weights[i].T * self._dactivate(self._as[i])
            self._dweights[i-1] = self._as[i-1].T @ self._dzs[i-1] / batch_size
            self._dbiases[i-1] = np.mean(self._dzs[i-1], axis=0)
            
    
    def step(self, lr, l2_reg):
        """
        Update weights and biases
        """
        for i in range(self._num_hidden_layers+1):
            self._weights[i] -= lr * (self._dweights[i] + l2_reg * self._weights[i])
            self._biases[i] -= lr * self._dbiases[i]
            
            
    def plot_weights(self, file_prefix, dataset):
        """
        Plot weights
        """
        
        # reshape multiplied weights
        weights = self._weights[0] + self._biases[0]
        for i in range(1, self._num_hidden_layers+1):
            weights = weights @ self._weights[i] + self._biases[i]
        weights = weights.reshape(int(np.sqrt(self._input_size)), int(np.sqrt(self._input_size)), self._output_size)
        
        # plot weights and samples for each class
        plt.figure(figsize=(10, 15))
        for i in range(self._output_size):
            plt.subplot(self._output_size, 8, 8*i+1)
            
            # plot weights
            plt.imshow(weights[:, :, i], vmin=0)
            plt.title(f'Class {i} weights')
            plt.axis('off')
            
            # plot samples
            images, labels = dataset.get_all_data()
            samples = images[labels == i]
            for j in range(7):
                plt.subplot(self._output_size, 8, 8*i+j+2)
                plt.imshow(samples[j].reshape(28, 28), cmap='gray')
                plt.axis('off')
            
        # save
        plt.tight_layout()
        plt.savefig(f'{file_prefix}_weights_samples.png')
