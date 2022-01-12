import argparse
import os
from models import VAE, vae_loss_function
from torch.optim import Adam
from utils import *
import torch
import matplotlib.pyplot as plt

# Instantiate the parser
parser = argparse.ArgumentParser(description='Vanilla VAE')

parser.add_argument('results_folder')

parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--latent_dim', type=int, default=64)

parser.add_argument('--generative_model', type=str, default='VanillaVAE')

parser.add_argument('--dataset', type=str, default='mnist')

parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--epoch', type=int, default=50)

args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_show = 10

if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)
    
generated_images_folder = os.path.join(args.results_folder, 'generated_images')
if not os.path.exists(generated_images_folder):
    os.makedirs(generated_images_folder)

X_train, y_train, X_test, y_test = read_data_flattened(args.dataset)
input_dim = X_train.shape[1]

if args.generative_model == 'VanillaVAE':
    model = VAE(input_dim, args.latent_dim, DEVICE).to(DEVICE)
    loss_function = vae_loss_function
    
    
optimizer = Adam(model.parameters(), lr=args.learning_rate)

print("Start training {model_name}...".format(model_name=args.generative_model))

train_size = X_train.shape[0]

for epoch in range(args.epoch):
    model.train()
    
    training_loss = 0
    
    train_indices = np.arange(0, train_size)
    np.random.shuffle(train_indices)
    
    for start in range(0, train_size, args.batch_size): 
        end = start + args.batch_size
        if (end > train_size):
            end = train_size
        
        X_batch = torch.from_numpy(X_train[train_indices[start:end]]).to(DEVICE)
           
        optimizer.zero_grad()

        x_hat, mean, log_var = model(X_batch)
        loss = loss_function(X_batch, x_hat, mean, log_var)
        
        training_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", training_loss / train_size)
    
    gener = x_hat[:n_show].detach().numpy().reshape(n_show, 28, 28)
    fig, ax = plt.subplots(1, n_show, figsize=(15,5))
    for i in range(n_show):
        ax[i].imshow(gener[i], cmap='gray')
        ax[i].axis('off')
#     plt.show()
    plt.savefig(os.path.join(generated_images_folder, 'epoch' + str(epoch) + '.png'))
    plt.close(fig)

X_train_latent_path = get_latent_path(args.results_folder, args.generative_model, 'train')
X_test_latent_path = get_latent_path(args.results_folder, args.generative_model, 'test')

save_latent_vector(model.encoder, X_train_latent_path, X_train, args.latent_dim)
save_latent_vector(model.encoder, X_test_latent_path, X_test, args.latent_dim)

