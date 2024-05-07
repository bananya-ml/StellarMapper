import torch
import torch.nn as nn
import numpy as np
from .networks.base_networks import get_network
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel
import os
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

class StarcNet(BaseModel):
    '''
        Parameters
        -----------
        num_fluxes : int
            Number of fluxes in the spectrum.
        num_filters : list of int
            Number of filters in each convolutional layer.
        filter_length : int
            Length of the convolutional filters.
        pool_length : int
            Length of the pooling filters.
        num_hidden : list of int
            Number of hidden units in each fully connected layer.
        num_labels : int
            Number of labels to predict. 
        device : str
            Device to run the model on. Either 'cpu' or 'cuda'.

        Attributes
        -----------

        '''

    def __init__(self, num_fluxes=None, num_filters=[4, 16], learning_rate=1e-3, network='StarNet', 
                 filter_length=8, batch_size=32, pool_length=4, num_hidden=[256, 128], 
                 num_labels=3, device='cuda', sampler=None, prt_interval=200, weight_decay=0.0,
                 val_split = 0.8, epochs=100, random_state=42):
        super(StarcNet, self).__init__(model_name='StarNet', learning_rate=learning_rate, network=network,
                                      batch_size=batch_size, device=device, prt_interval=prt_interval,
                                      weight_decay=weight_decay, val_split=val_split, epochs=epochs, 
                                      random_state=random_state
                                      )
        self.num_fluxes = num_fluxes
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.pool_length = pool_length
        self.num_hidden = num_hidden
        self.num_labels = num_labels
        self.val_split = val_split
        self.weight_decay = weight_decay

        self.network = network

        self.sampler = sampler
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.random_state = random_state


    def training_prepare(self, X, y):
        
        train_size = int(len(X) * self.val_split)

        X_train, X_val = np.split(X, [train_size])
        y_train, y_val = np.split(y, [train_size])

        train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                torch.from_numpy(y_train).float())
        
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
                                torch.from_numpy(y_val).float())
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=self.sampler, shuffle=True if self.sampler is None else False)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, sampler=self.sampler,shuffle=True if self.sampler is None else False)
        
        criterion = nn.MSELoss()

        network_class = get_network(self.network)

        network_params = {
            'num_fluxes': self.num_fluxes,
            'num_filters': self.num_filters,
            'filter_length': self.filter_length,
            'pool_length': self.pool_length,
            'num_hidden': self.num_hidden,
            'num_labels': self.num_labels,
        }

        net = network_class(**network_params).to(self.device)
        print(net)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return train_loader, val_loader, net, criterion, optimizer

    def inference_prepare(self, X, y):
        '''
        '''
        if y is not None:
            dataset = TensorDataset(torch.from_numpy(X).float().to(self.device),
                                    torch.from_numpy(y).float().to(self.device))
        else:
            dataset = TensorDataset(
                torch.from_numpy(X).float().to(self.device))

        test_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        # self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):

        batch_x, batch_y = batch_x

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        z = net(batch_x.unsqueeze(1))

        loss = criterion(z, batch_y)

        return loss

    def inference_forward(self, batch_x, batch_y, net, criterion):

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device) if batch_y is not None else None

        y_pred = net(batch_x.unsqueeze(1))

        if batch_y is not None:
            loss = criterion(y_pred, batch_y)
        else:
            loss = None

        return y_pred, loss
   
    def save_info(self, path):
        '''
        
        '''
        model_info = {
            'input_dims': self.num_fluxes,
            'num_filters': self.num_filters,
            'filter_length': self.filter_length,
            'pool_length': self.pool_length,
            'num_hidden': self.num_hidden,
            'num_labels': self.num_labels,
        }

        torch.save(model_info, os.path.join(path))

    def load_info(self, path):
        '''
        
        '''

        model_info = torch.load(os.path.join(path, self.model_name + '_model_info.pth'))

        self.num_fluxes = model_info['input_dims']
        self.num_filters = model_info['num_filters']
        self.filter_length = model_info['filter_length']
        self.pool_length = model_info['pool_length']
        self.num_hidden = model_info['num_hidden']
        self.num_labels = model_info['num_labels']

        network_params = {
            'num_fluxes': self.num_fluxes,
            'num_filters': self.num_filters,
            'filter_length': self.filter_length,
            'pool_length': self.pool_length,
            'num_hidden': self.num_hidden,
            'num_labels': self.num_labels,
        }

        self.net = get_network(self.network)(**network_params).to(self.device)
        
        return self.net

    def save_model(self, save_path):
        """
        Save the trained model to a file.

        Parameters
        ----------
        save_path : str
            The path where the model will be saved.
        """
        super().save_model(save_path)

    def load_model(self, load_path):
        """
        Load a pre-trained model and its relevant information.

        Parameters
        ----------
        load_path : str
            The path from which the model will be loaded.
        """
        super().load_model(load_path)