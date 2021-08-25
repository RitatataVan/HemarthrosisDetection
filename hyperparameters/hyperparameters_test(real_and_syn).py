"""
Hyperparameters for a run.
"""

parameters = {
    # Random Seed
    'seed': 123,

    # Data
    'train_data': '../datasets/train_real_100_syn_0.csv',  # Path to the training iamges directory


    'test_data': '../datasets/testing_data.csv',  # Path to the training labels director0nano

    'img_size': 380,  # Image input size (this might change depending on the model)  # was 380
    'batch_size': 25,  # Input batch size for training (you can change this depending on your GPU ram)
    'data_mean': [0.1129, 0.1157, 0.1180],  # Mean values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'data_std': [0.1546, 0.1575, 0.1595],  # Std Dev values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'out_features': 1,  # For binary is 1

    # Model
    'model': 'efficientnet',  # Model to train (This name has to correspond to a model from models.py)
    'optimizer': 'ADAM',  # Optimizer to update model weights (Currently supported: ADAM or SGD)
    'criterion': 'BCEWithLogitsLoss',
    'lear_rate': 0.001,  # Learning Rate to use
    'min_epochs': 1,  # Minimum number of epochs to train for
    'epochs': 50,  # Number of epochs to train for
    'precision': 16,  # Pytorch precision in bits
    'accumulate_grad_batches': 2,  # the number of batches to estimate the gradient from
    'num_workers': 0  # Number of CPU workers to preload the dataset in parallel
}

print(parameters['train_data'])
