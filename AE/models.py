import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import collections
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from AE.utility import create_activation,compute_distance_matrix
sys.path.append('../')

class MLP(nn.Module):
    def __init__(self, layers_list, dropout, norm,activation,last_act=False):
        super(MLP, self).__init__()
        layers=nn.ModuleList()
        assert len(layers_list)>=2, 'no enough layers'
        for i in range(len(layers_list)-2):
            layers.append(nn.Linear(layers_list[i],layers_list[i+1]))
            if norm:
                layers.append(nn.BatchNorm1d(layers_list[i+1]))
            if activation is not None:
                layers.append(activation)
            if dropout>0.:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layers_list[-2],layers_list[-1]))
        if norm:
            layers.append(nn.BatchNorm1d(layers_list[-1]))
        if last_act:
            if activation is not None:
                layers.append(activation)
        if dropout>0.:
            layers.append(nn.Dropout(dropout))
        # layers.append(nn.Linear(layers_list[-1],out_dim))
        self.network = nn.Sequential(*layers)
        # self.apply(init_weights_xavier_uniform)
    def forward(self,x):
        for layer in self.network:
            x=layer(x)
        return x
class LatentModel(nn.Module):
    def __init__(self,n_hidden,n_latent,
                 kl_weight=1e-6, warmup_step=10000):
        super(LatentModel,self).__init__()
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        # self.kl = 0
        self.kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step

    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = 0.0
        else:
            self.kl_weight = self.kl_weight + (1e-2 - 1e-6) / self.warmup_step

        # elif self.step_count == self.warmup_step:
        #     pass
            # self.step_count = 0
            # self.kl_weight = 1e-6

    def forward(self, h):
        mu = self.mu(h)
        log_var = self.logvar(h)
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        if self.training:
            z = mu + sigma * epsilon
            # (1 + log_var - mu ** 2 - log_var.exp()).sum()* self.kl_weight
            # print('hhhhhhh')
            self.kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum()  * self.kl_weight#/ z.shape[0]
            self.kl_schedule_step()
        else:
            z = mu
        return z





class AutoEncoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            n_layers: int = 1,
            n_hidden: int = 500,
            n_latent: int = 10,
            activate_type: str='relu',
            dropout: float = 0.2,
            norm: bool = False,
            seed: int=42,
    ):
        '''
        Autoencoder model.
        Encoder and Decoder take identical architectures.

        Parameters:
            in_dim:
                dimension of the input feature
            n_layers:
                number of hidden layers
            n_hidden:
                dimension of hidden layer. All hidden layers take the same dimensions
            n_latent:
                dimension of latent space
            activate_type:
                activation functions.
                Options: 'leakyrelu','relu', 'gelu', 'prelu', 'elu', and None for identity map.
            dropout:
                dropout rate
            norm:
                whether to include batch normalization layer
            seed:
                random seed.
        '''
        super(AutoEncoder,self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.in_dim=in_dim
        self.n_layers=n_layers
        self.n_hidden=n_hidden
        self.n_latent=n_latent
        self.dropout=dropout
        self.norm=norm
        self.activation=create_activation(activate_type)
        ## Encoder:
        self.encoder_layer=[in_dim]
        for i in range(n_layers):
            self.encoder_layer.append(n_hidden)
        self.encoder=MLP(self.encoder_layer,dropout,norm,self.activation,last_act=True)
        self.encoder_to_latent=MLP([self.encoder_layer[-1],n_latent],
                                   dropout,norm,self.activation)

        ## Decoder:
        self.decoder_layer=[n_latent]
        for i in range(n_layers):
            self.decoder_layer.append(n_hidden)
        self.decoder=MLP(self.decoder_layer,dropout,norm,self.activation,last_act=True)
        self.decoder_to_output=MLP([self.decoder_layer[-1],self.in_dim],dropout,norm,activation=None)


    def forward(self,x):
        rep=self.get_latent_representation(x,tensor=True)
        h = self.decoder(rep)
        x_recon=self.decoder_to_output(h)
        mse = nn.MSELoss(reduction='sum')
        loss = mse(x_recon, x)/x.shape[1]
        return loss


    def get_latent_representation(self,x,tensor:bool=False):
        '''
        Get latent space representation

        Parameters
        x:
            Input space
        tensor:
            If input x is a tensor, or it is a numpy array
        Return
        rep:
            latent space representation
            If tensor==True:
                return a tensor
            If tensor==Flase:
                return a numpy array
        '''
#        if not tensor:
#            x=torch.tensor(x,dtype=torch.float32)
#            self.eval()
        x=self.encoder(x)
        rep=self.encoder_to_latent(x)
        #if not tensor:
        #    rep=rep.detach().numpy()
        return rep
    def get_reconstruction(self, x):
        '''
        Reconstruct/impute gene expression data
        x:
            features. Numpy array
        Return
        x_recon:
            Numpy array
        '''
        self.eval()
        x=torch.tensor(x,dtype=torch.float32)
#        with torch.no_grad():
        x=self.encoder(x)
        x=self.encoder_to_latent(x)
        x = self.decoder(x)
        x_recon = self.decoder_to_output(x)

        #x_recon=x_recon.detach().numpy()
        return x_recon
    def get_generative(self,z):
        '''
        genereate gene expression data from latent space variable
        z:
            latent space representation. Numpy array
        Return
        x_recon:
            Numpy array
        '''
        self.eval()
        #z=torch.tensor(z,dtype=torch.float32)
#        with torch.no_grad():
        x = self.decoder(z)
        x_recon = self.decoder_to_output(x)
        #x_recon=x_recon.detach().numpy()
        return x_recon




