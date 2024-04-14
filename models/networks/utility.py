import numpy as np
import warnings
import importlib
from torch import nn
import torch

def _instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()


def _handle_n_hidden(n_hidden):
    if type(n_hidden) == int:
        n_layers = 1
        hidden_dim = n_hidden
    elif type(n_hidden) == str:
        n_hidden = n_hidden.split(',')
        n_hidden = [int(a) for a in n_hidden]
        n_layers = len(n_hidden)
        hidden_dim = int(n_hidden[0])

        if np.std(n_hidden) != 0:
            warnings.warn('use the first hidden num, '
                          'the rest hidden numbers are deprecated', UserWarning)
    else:
        raise TypeError('n_hidden should be a string or a int.')

    return hidden_dim, n_layers

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def _compute_out_size(in_size, mod):
        """
        Compute output size of Module `mod` given an input with size `in_size`.
        """
        f = mod.forward(torch.autograd.Variable(torch.Tensor(1, *in_size)))
        return f.size()[1:]