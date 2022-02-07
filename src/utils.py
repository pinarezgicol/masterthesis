from keras.datasets import mnist
import numpy as np
import os
import torch
from medmnist import INFO, Evaluator
import dataset_without_pytorch
import random

def read_data(dataset):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == "organamnist":
        info = INFO[dataset]
        # task = info['task']
        # n_channels = info['n_channels']
        # n_classes = len(info['label'])
        DataClass = getattr(dataset_without_pytorch, info['python_class'])

        train_dataset = DataClass(split='train', download=True)
        X_train = train_dataset.imgs
        y_train = np.asarray([label[0] for label in train_dataset.labels])

        test_dataset = DataClass(split='test', download=True)
        X_test = test_dataset.imgs
        y_test = np.asarray([label[0] for label in test_dataset.labels])


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32') / 255.
    
    return X_train, y_train, X_test, y_test

def delete_elements(X, y, number_of_elements, label):
    indices = np.where(y == label)[0]
    np.random.shuffle(indices)

    X = np.delete(X, indices[number_of_elements:], axis=0)
    y = np.delete(y, indices[number_of_elements:], axis=0)

    return X, y


def make_imbalanced_data(dataset, X_train, y_train):
    if dataset == "mnist":
        X_train, y_train = delete_elements(X_train, y_train, 500, 0)

        X_train, y_train = delete_elements(X_train, y_train, 500, 3)

        X_train, y_train = delete_elements(X_train, y_train, 500, 9)

    if (dataset == "organamnist"):
        X_train, y_train = delete_elements(X_train, y_train, 380, 5)

        X_train, y_train = delete_elements(X_train, y_train, 140, 1)

    return X_train, y_train

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)