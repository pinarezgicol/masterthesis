from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, latent_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),)
        
        
    def forward(self, x):
        return self.layers(x)
    
class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var      
    
class Decoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()

        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        h = self.relu(self.fc4(z))
        h = self.relu(self.fc5(h))
        return self.sigmoid(self.fc6(h))     

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device):     
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.device = device
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu) # return z sample
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.input_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD