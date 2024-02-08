import torch
from torch import nn
from abc import ABC, abstractmethod
import time
import numpy as np
import random
from tqdm import tqdm


class BaseModel(ABC, nn.Module):
    '''
    '''

    def __init__(self, num_fluxes, num_filters, learning_rate,
                 network, filter_length, batch_size, pool_length, num_hidden,
                 num_labels, device, epochs, random_state):
        super(BaseModel, self).__init__()

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

        self.set_seed(self.random_state)
        return

    def fit(self, X, y):
        '''
        '''


        print("Starting training...")

        self.train_data = np.array(X)
        self.train_label = np.array(y)
        
        self.train_loader, self.net, self.criterion = self.training_prepare(
            self.train_data, y=self.train_label)
        self._training()
        return

    def predict(self, X, y=None):
        '''
        '''

        self.val_data = np.array(X)
        self.val_label = np.array(y) if y is not None else None

        if self.val_label is not None:
            print("Starting inference on validation data...")
            self.test_loader = self.inference_prepare(
                self.val_data, self.val_label)

            loss = self._validate()

            return loss

        else:
            print("Starting inference on test data...")
            self.test_loader = self.inference_prepare(self.val_data, y=None)

            y_pred = self._inference()

            return y_pred

    def _training(self):
        '''
        '''
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate)

        self.net.train()
        for epoch in range(self.epochs):
            t1 = time.time()
            total_loss = 0
            cnt = 0

            for batch_x in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch'):
                loss = self.training_forward(batch_x, self.net, self.criterion)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

            t = time.time()-t1
            print("Epoch: {:3d},  training loss: {:.6f}, time: {:.1f} seconds".format(
                epoch+1, total_loss/len(self.train_loader), t))
        return

    def _inference(self):
        '''
        '''

        self.net.eval()
        for epoch in range(self.epochs):
            t1 = time.time()

            with torch.inference_mode():

                for i, data in enumerate(tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch'), 0):
                    batch_x, batch_y = data, None
                    y_pred, loss = self.inference_forward(
                        batch_x[0], batch_y, self.net, self.criterion)

            t = time.time()-t1
            print('Epoch: {:3d}, time:{:.1f} seconds'.format(
                epoch+1, t))

        return y_pred

    def _validate(self):
        '''
        '''
        self.net.eval()
        for epoch in range(self.epochs):
            t1 = time.time()
            total_loss = 0

            with torch.inference_mode():

                for i, data in enumerate(tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch'), 0):
                    batch_x, batch_y = data
                    y_pred, loss = self.inference_forward(
                        batch_x, batch_y, self.net, self.criterion)
                    total_loss += loss.item()

            val_loss = total_loss/len(self.test_loader)
            t = time.time()-t1
            print('Epoch: {:3d}, time:{:.1f} seconds'.format(
                epoch+1, t))
        print('Validation loss: {:.6f}'.format(val_loss))

        return val_loss

    @abstractmethod
    def training_prepare(self, X, y):
        pass

    @abstractmethod
    def training_forward(self, x):
        pass

    @abstractmethod
    def inference_forward(self, X, y):
        pass

    @abstractmethod
    def inference_prepare(self, X):
        pass

    @abstractmethod
    def inference_forward(self, X):
        pass

    @staticmethod
    def set_seed(seed):
        '''
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True

        return
