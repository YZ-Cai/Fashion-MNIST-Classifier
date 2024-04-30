import numpy as np


class Model:
    def __init__(self, input_size, hidden_sizes, activation_type, output_size):
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_hidden_layers = len(hidden_sizes)
        self._output_size = output_size
        self._activation_type = activation_type
        self._init_params()
        
        
    def _init_params(self):
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
        
        
    def _activate(self, X):
        """
        Activation function
        """
        if self._activation_type == 'relu':
            return np.maximum(X, 0)
        elif self._activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self._activation_type == 'tanh':
            return np.tanh(X)
        else:
            raise ValueError(f'Invalid activation type: {self._activation_type}')
        
        
    def _dactivate(self, X):
        """
        Derivative of activation function
        """
        if self._activation_type == 'relu':
            return (X > 0).astype(float)
        elif self._activation_type == 'sigmoid':
            return X * (1 - X)
        elif self._activation_type == 'tanh':
            return 1 - X**2
        else:
            raise ValueError(f'Invalid activation type: {self._activation_type}')
        
        
    def _softmax(self, X):
        """
        Softmax function
        """
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
        
        
    def forward(self, X):
        """
        Forward pass through the network
        """
        self._zs = []
        self._as = [X]
        for i in range(self._num_hidden_layers):
            self._zs.append(self._as[-1] @ self._weights[i] + self._biases[i])
            self._as.append(self._activate(self._zs[-1]))
        self._zs.append(self._as[-1] @ self._weights[-1] + self._biases[-1])
        y_pred = self._softmax(self._zs[-1])
        return y_pred
        
        
    def backward(self, labels, y_pred):
        """
        Backward pass through the network
        """
        
        # init gradients
        self._dzs = [None for z in self._zs]
        self._dweights = [None for w in self._weights]
        self._dbiases = [None for b in self._biases]
        
        # one-hot encoding of labels
        y = np.zeros((labels.size, 10))
        y[np.arange(labels.size), labels] = 1
        batch_size = y.shape[0]
        
        # calculate gradients
        self._dzs[-1] = y_pred - y
        self._dweights[-1] = self._as[-1].T @ self._dzs[-1] / batch_size
        self._dbiases[-1] = np.mean(self._dzs[-1], axis=0)
        for i in range(self._num_hidden_layers, 0, -1):
            self._dzs[i-1] = self._dzs[i] @ self._weights[i].T * self._dactivate(self._zs[i-1])
            self._dweights[i-1] = self._as[i-1].T @ self._dzs[i-1] / batch_size
            self._dbiases[i-1] = np.mean(self._dzs[i-1], axis=0)
            
    
    def step(self, lr, l2_reg):
        """
        Update weights and biases
        """
        for i in range(self._num_hidden_layers+1):
            self._weights[i] -= lr * (self._dweights[i] + l2_reg * self._weights[i])
            self._biases[i] -= lr * self._dbiases[i]