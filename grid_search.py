import numpy as np
import pandas as pd
from core.utils import Dataset
from core.model import Model
from core.train import train
from core.test import test



# set random seeds
np.random.seed(0)

# configs
param_grid = {
    'hidden_size_1': [128, 512, 1024],
    'hidden_size_2': [32, 64, 128],
    'init_lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'l2_reg': [0.01, 0.001, 0.0001, 0.00001],
}

# Load fashion MNIST dataset
train_dataset = Dataset(kind='train')
val_images, val_labels = train_dataset.split_validation_data(train_ratio=0.95)
test_images, test_labels = Dataset(kind='t10k').get_all_data()

# grid search
best_val_accuracy = 0
best_params = None
df_results = pd.DataFrame()
for hidden_size_1 in param_grid['hidden_size_1']:
    for hidden_size_2 in param_grid['hidden_size_2']:
        for init_lr in param_grid['init_lr']:
            for l2_reg in param_grid['l2_reg']:
                hidden_sizes = [hidden_size_1, hidden_size_2]
                file_prefix = f'./outputs/hidden{hidden_sizes}_lr{init_lr}_l2{l2_reg}'
                
                # initialize model
                print('='*60+f'\nHidden sizes: {hidden_sizes}, Learning rate: {init_lr}, L2 reg: {l2_reg}')
                file_prefix = f'./outputs/hidden{hidden_sizes}_lr{init_lr}_gamma0.95_l2{l2_reg}'
                model = Model(train_dataset.x_dim, hidden_sizes, train_dataset.y_dim, activation_type='relu')
                model.init_params()
                
                # train the model
                train(train_dataset, model, val_images, val_labels, file_prefix, init_lr=init_lr, l2_reg=l2_reg)
                
                # validation and test accuracy
                model.load(file_prefix+'.pkl')
                val_loss, val_accuracy = test(val_images, val_labels, model)
                test_loss, test_accuracy = test(test_images, test_labels, model)
                
                # write to result
                df_results = pd.concat([df_results, pd.DataFrame([{
                    'hidden_size_1': hidden_size_1, 'hidden_size_2': hidden_size_2, 'init_lr': init_lr, 'l2_reg': l2_reg,
                    'val_loss': val_loss, 'val_accuracy': val_accuracy, 'test_loss': test_loss, 'test_accuracy': test_accuracy
                }])], ignore_index=True)
                df_results.to_csv('./outputs/grid_search_results.csv', index=False)
                
                # update best parameters
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {'hidden_size_1': hidden_size_1, 'hidden_size_2': hidden_size_2, 'init_lr': init_lr, 'l2_reg': l2_reg}
                print(f'\nCurrent best validation accuracy: {best_val_accuracy*100:.1f}%')
                print(f'Current best parameters: {best_params}')
