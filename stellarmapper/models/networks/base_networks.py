import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Union, cast, Dict
from functools import reduce
import operator
from .utility import _instantiate_class, _handle_n_hidden, _init_weights, _compute_out_size

net_name = ['StarNet', 'MLPnet', 'ConvSeqEncoder', 'StarcNet', 'OTrain', 'AlexNet', 'VGGNet' ]

def get_network(network_name):
    map = {
        'StarNet': StarNet,
        'MLP': MLPnet,
        'StarcNet': StarcNet,
        'OTRAIN': OTrain,
        'AlexNet': AlexNet,
        'VGG': VGGNet
    }

    if network_name in map.keys():
        return map[network_name]
    else:
        raise NotImplementedError(f"Network {network_name} not found!"
                                  f" Available networks: {map.keys()}")

class StarNet(nn.Module):
    '''
    StarNet network constructed from "An application of deep learning in the analysis of stellar spectra"
    '''
    def __init__(self, num_fluxes=None, num_filters=[4, 16], filter_length=8,
                 pool_length=4, num_hidden=[256, 128], num_labels=3):
        super(StarNet, self).__init__()

        self.num_labels = num_labels

        self.conv1 = nn.Conv1d(1, num_filters[0], filter_length)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], filter_length)
        self.pool = nn.MaxPool1d(pool_length, pool_length)

        pool_output_shape = _compute_out_size((1, num_fluxes), nn.Sequential(self.conv1, 
                                                                             self.conv2, 
                                                                             self.pool))

        self.fc1 = nn.Linear(
            pool_output_shape[0]*pool_output_shape[1], num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.output = nn.Linear(num_hidden[1], num_labels)

        self.apply(_init_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        if self.num_labels == 1:
            x = F.relu(x)
        return x

class StarcNet(nn.Module):
    '''
    StarcNet network constructed from "STARCNET: Machine Learning for Star Cluster Identification"
    '''
    def __init__(self, input_dim = 5, size = 32, n=8, image_size = 32, mag_dim = 10):
        super(StarcNet, self).__init__()
        self.a1 = 1 # filters multiplier
        self.a21 = 1
        self.a2 = 1
        self.a3 = 1
        n = 8 # for groupnorm
        self.sz1 = image_size # size of input image (sz1 x sz1)
        self.sz = mag_dim # for secon input size (2*sz x 2*sz) default: 10
        
        self.resize = nn.Upsample(size=(self.sz1,self.sz1))
        
        self.net = nn.Sequential(ConvBlock(input_dim, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n))

        self.outblock = OutBlock(size, n)
        self.fc = nn.Linear(384, 4)

        self.apply(_init_weights)
        
    def forward(self, x):
        x0 = x[:,:,:,:]#.unsqueeze(1)
        x1 = x[:,:,int(self.sz1/2-self.sz):int(self.sz1/2+self.sz), int(self.sz1/2-self.sz):int(self.sz1/2+self.sz)] #.unsqueeze(1)
        x2 = x[:,:,int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1), int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1)]#.unsqueeze(1)
        x1 = self.resize(x1)
        x2 = self.resize(x2)

        # first conv net
        out0 = self.net(x0)
        out0 = self.outblock(out0.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
        
        # second conv net
        out1 = self.net(x1)
        out1 = self.outblock(out1.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
         
        # third conv net
        out2 = self.net(x2)
        out2 = self.outblock(out2.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
        
        # combine all 3 outputs to make a single prediction
        out = torch.cat((out0,out1,out2),1)
        out = self.fc(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(n, out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))
    
class OutBlock(nn.Module):
    def __init__(self, size, n):
        super().__init__()
        self.fc = nn.Linear(128*int(size/2)*int(size/2), 128)
        self.gn = nn.GroupNorm(n,128)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        return self.dropout(self.relu(self.gn(self.fc(x))))
    
class MLPnet(nn.Module):
    def __init__(self, n_features, n_hidden='500,100', n_output=20, mid_channels=None,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        # for only use one kind of activation layer
        if type(activation) == str:
            activation = [activation] * num_layers
            activation.append(None)

        assert len(activation) == len(n_hidden)+1, 'activation and n_hidden are not matched'

        self.layers = []
        for i in range(num_layers+1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            mid_channels=mid_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation[i],
                            skip_connection=skip_connection if i != num_layers else 0,
                            dropout=dropout if i !=num_layers else None)
            ]
        self.network = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_output, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i])+n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels
    
class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 activation='Tanh', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # Tanh, ReLU, LeakyReLU, Sigmoid
        if activation is not None:
            self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.batch_norm = batch_norm
        if batch_norm is True:
            dim = out_channels if mid_channels is None else mid_channels
            self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1
     
class OTrain(nn.Module):
    '''
    OTRAIN network constructed from "O’TRAIN: A robust and flexible ‘real or bogus’ classifier for thestudy of the optical transient sky"
    '''
    def __init__(self, num_channels=1, size=(32,32), num_classes=2, dropout=0.3, hidden_dims = [512,256], filter_size = (3,3), pool_size = (2,2)):
        super(OTrain, self).__init__()
        

        self.conv1 = nn.Conv2d(num_channels, 16, filter_size)
        self.conv2 = nn.Conv2d(16, 32, filter_size)
        self.pool1 = nn.AvgPool2d(pool_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv3 = nn.Conv2d(32, 64, filter_size)
        self.pool2 = nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.conv4 = nn.Conv2d(64, 128, filter_size)
        self.conv5 = nn.Conv2d(128, 256, filter_size)
        self.pool3 = nn.MaxPool2d(pool_size)

        
        output_shape = _compute_out_size((num_channels, *(size)),nn.Sequential(self.conv1, self.conv2, 
                                                                             self.pool1, self.dropout1, 
                                                                             self.conv3, self.pool2, 
                                                                             self.dropout2, self.conv4, 
                                                                             self.conv5, self.pool3))
        
        self.fc1 = nn.Linear(reduce(operator.mul, output_shape), hidden_dims[0])
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        
        self.apply(_init_weights)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(self.pool2(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        
        x = F.softmax(self.fc3(self.fc2(self.dropout3(self.fc1(x)))))
        
        return x

class AlexNet(nn.Module):
    '''
    AlexNet network constructed from "One weird trick for parallelizing convolutional neural networks"
    '''
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGGNet(nn.Module):
    '''
    VGGNet constructed from "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    '''
    def __init__(
        self, config: str = 'A', num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, batch_norm: bool = False):
        super().__init__()

        cfg = self._config(config)
        features = self._make_layers(cfg, batch_norm)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _config(self, config):
        cfgs: Dict[str, List[Union[str, int]]] = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        return cfgs[config]

    def _make_layers(self, cfg: List[Union[str, int]], batch_norm: bool = False):        
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
            