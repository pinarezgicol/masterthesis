import argparse
import os
import torch
from acquisition_functions import *
from utils import *
from models import MLP
from skorch import NeuralNetClassifier
from al import *


# Instantiate the parser
parser = argparse.ArgumentParser(description='DBAL Algorithm')

parser.add_argument('results_folder', type=str)

parser.add_argument('dataset', type=str)

parser.add_argument('--latent_size', type=int, default=64)

parser.add_argument('--X_train_path', type=str, default=None)

parser.add_argument('--X_test_path', type=str, default=None)

parser.add_argument('--max_epochs', type=int, default=200)

parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--experiment_count', type=int, default=3)

args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)

ACQ_FUNCS = {
    "var_ratios": var_ratios,
    "mean_std": mean_std,
    "max_entropy": max_entropy,
    "bald": bald,
    "uniform": uniform
}

X_train, y_train, X_test, y_test = read_data(args.dataset)
latent_size = 784

if args.X_train_path and args.X_test_path:
    X_train = np.load(args.X_train_path)
    X_test = np.load(args.X_test_path)    
    latent_size = args.latent_size
    print('GEN-DBAL running')
else:
    print('DBAL running')

number_of_classes = len(np.unique(y_train))
    
for exp_iter in range(args.experiment_count):
    np.random.seed(exp_iter)
    initial_idx = np.array([],dtype=int)
    for i in range(number_of_classes):
        idx = np.random.choice(np.where(y_train==i)[0], size=2, replace=False)
        initial_idx = np.concatenate((initial_idx, idx))
        
    for func_name, acquisition_func in ACQ_FUNCS.items():  
        X_initial = X_train[initial_idx]
        y_initial = y_train[initial_idx]

        X_pool = np.delete(X_train, initial_idx, axis=0)
        y_pool = np.delete(y_train, initial_idx, axis=0)

        model = MLP(latent_size).to(DEVICE)

        estimator = NeuralNetClassifier(model,
                                        max_epochs=args.max_epochs,
                                        batch_size=args.batch_size,
                                        lr=args.learning_rate,
                                        optimizer=torch.optim.Adam,
                                        criterion=torch.nn.CrossEntropyLoss,
                                        train_split=None,
                                        verbose=0,
                                        device=DEVICE)



        acc_arr, dataset_size_arr = active_learning_procedure(acquisition_func,
                                                              X_test,
                                                              y_test,
                                                              X_pool,
                                                              y_pool,
                                                              X_initial,
                                                              y_initial,
                                                              estimator,)
        
        file_name = os.path.join(args.results_folder, func_name + "_exp_" + str(exp_iter) + ".npy")
        np.save(file_name, (acc_arr, dataset_size_arr))