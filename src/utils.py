from keras.datasets import mnist
import numpy as np
import os
import torch

def read_data(dataset):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32') / 255.
    
    return X_train, y_train, X_test, y_test


def read_data_flattened(dataset):
    X_train, y_train, X_test, y_test = read_data(dataset)
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test
    

def save_latent_vector(model, results_path, X, latent_dim):
    model.eval()

    X_latent = np.zeros((X.shape[0], latent_dim), dtype="float32")

    for ind, x_point in enumerate(X):
        z_point, _ = model(torch.from_numpy(x_point))
        X_latent[ind] = z_point.detach().numpy()
       
    np.save(results_path, X_latent)
    
def get_latent_path(path, model_type, dataset_type):
    return os.path.join(path, model_type + '_{dataset_type}.npy'.format(dataset_type=dataset_type))