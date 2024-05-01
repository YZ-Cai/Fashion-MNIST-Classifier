# ImageClassifier

Codes for the first homework for machine learning course DATA620004, School of Data Science, Fudan University.
Please follow the following instructions to run the codes.

## Prepare dataset

In the path `data/`, run `download.sh` to obtain the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## Train and test

To train and test the model under certain hyper parameters, modify the configurations in `main.py` if needed:

```python
# configs
hidden_sizes = [512, 128]
activation_type = 'relu'
init_lr=1e-3
gamma=0.95
l2_reg=1e-3
train_ratio = 0.95
```

The above parameters are described as follows:

- `hidden_sizes`: the number and sizes of hidden layers. For example, `[512, 128]` means there are two hidden layers with sizes 512 and 128, respectively.
- `activation_type`: the activation function used in hidden layers, `relu`, `sigmoid` and `tanh` are supported.
- `init_lr`: specify the initial learning rate
- `gamma`: the parameter for exponential decay of learning rate during training, which is in (0, 1). Larger value results in faster decrease of learning rate.
- `l2_reg`: the parameter for L2 regularization during training. 
- `train_ratio`: the ratio of the training set. We use this parameter since the Fashion-MNIST dataset does not provide validation set, thus we should preserve some samples in the original training set for validation.

Use the following command to train and test a model:

```
python main.py
```

The resulting parameter file of the trained model appears in `./outputs/*.pkl`, and the training and validation loss and accuracy are saved in `./outputs/*.csv`.
The visualization of loss and accuracy curves, and the visualization of model parameters and samples from corresponding categories are saved in `./outputs/*.png`.


## Grid search for hyper parameters

The candidate hyper parameters are defined in `grid_search.py`:

```python
# configs
param_grid = {
    'hidden_size_1': [128, 512, 1024],
    'hidden_size_2': [32, 64, 128],
    'init_lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'l2_reg': [0.01, 0.001, 0.0001, 0.00001],
}
```

Use the following command to execute the hyper parameter search, in which the model will be trained and tested automatically:

```
python grid_search.py
```

The resulting validation and test accuracies along with corresponding hyper parameters are saved in `./outputs/grid_search_results.csv`.
