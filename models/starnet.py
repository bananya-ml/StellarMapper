import torch
import torch.nn as nn
from .networks.base_networks import get_network
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

class StarNet(BaseModel, nn.Module):
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

    def __init__(self, num_fluxes, num_filters=[4, 16], learning_rate=0.001, network='StarNet', filter_length=8, batch_size=32,
                 pool_length=4, num_hidden=[256, 128], num_labels=3, device='cuda',
                 epochs=100, random_state=42):
        super(StarNet, self).__init__(num_fluxes=num_fluxes, num_filters=num_filters, learning_rate=0.001, network='StarNet',
                                      filter_length=filter_length, batch_size=batch_size, pool_length=pool_length,
                                      num_hidden=num_hidden, num_labels=num_labels, device=device, epochs=epochs,
                                      random_state=random_state
                                      )
        self.num_fluxes = num_fluxes
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.pool_length = pool_length
        self.num_hidden = num_hidden
        self.num_labels = num_labels

        self.network = network

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.random_state = random_state

        return

    def training_prepare(self, X, y):

        

        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).unsqueeze(1).long())
        
        counter = Counter(y.flatten())

        weight_map = {label: 1./count for label, count in counter.items()}

        sampler = WeightedRandomSampler(weights=[weight_map[label.item()]for data,label in dataset],
                                num_samples=len(dataset),replacement=True)

        train_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, shuffle=True if sampler is None else False)

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

        return train_loader, net, criterion

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
