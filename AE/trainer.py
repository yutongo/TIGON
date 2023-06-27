import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import collections
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from AE.utility import create_activation

sys.path.append('../')
def dataloader_split(X,test_size,seed,batch_size):
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(torch.tensor(X_train,dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
class Trainer(object):
    def __init__(self,
                 model,
                 X,
                 test_size:float=0.1,
                 batch_size:int=32,
                 lr:float=1e-3,
                 weight_decay:float=0.0,
                 seed:int=42,):
        '''
        Trainer for pretrain model.
        Parameters:
        model:
            A pytorch model defined in "models.py"
        X:
            Feature matrix. mxn numpy array.
                a standarized logorithmic data (i.e., zero mean, unit variance)
        test_size:
            fraction of testing/validation data size. default: 0.2
        batch_size:
            batch size.
        lr:
            learning rate.
        weight_decay:
            L2 regularization.
        seed:
            random seed.
        '''
        # self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        # if self.model.decoder_type=='normal':
        self.train_loader,self.test_loader=\
            dataloader_split(X,test_size,seed,batch_size)
        self.seed=seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
    def train_step(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(self.train_loader):
            data = data.to(self.device)
            # scale_factor = scale_factor.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(data,)
            loss.backward()

            train_loss += loss.item()#*data.shape[0]
            self.optimizer.step()
        train_loss=train_loss / len(self.train_loader.dataset)

        return train_loss
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(self.test_loader):
                data = data.to(self.device)
                loss =self.model(data,)
                test_loss += loss.item()#*data.shape[0]
        test_loss /= len(self.test_loader.dataset)
        return test_loss
    def train(self, max_epoch=500,tol=1e-2,  patient=30):
        # self.model.train()
        self.model.history = {'train_loss': [], 'val_loss': [],
                              'train_loss_ae':[],'val_loss_ae':[],
                              'train_loss_topo':[],'val_loss_topo':[],
                              'epoch':[]}
        best_val_error = float('inf')
        num_patient_epochs = 0
        for epoch in range(max_epoch):
            self.epoch=epoch
            # Train for one epoch and get the training loss
            train_loss = self.train_step()
            # Compute the validation error
            val_loss = self.test()
            self.model.history['train_loss'].append(train_loss)
            self.model.history['val_loss'].append(val_loss)
            self.model.history['epoch'].append(epoch)
            # if epoch % 10==0:
            print(f"Epoch {epoch}: train loss = {train_loss:.4f}, val error = {val_loss:.4f}")
            # Check if the validation error has decreased by at least tol
            if best_val_error - val_loss >= tol:
                best_val_error = val_loss
                num_patient_epochs = 0
            else:
                num_patient_epochs += 1
                # Check if we have exceeded the patience threshold
            if num_patient_epochs >= patient:
                print(f"No improvement in validation error for {patient} epochs. Stopping early.")
                break
        print(f"Best validation error = {best_val_error:.4f}")
