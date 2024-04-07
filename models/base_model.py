import torch
from abc import ABCMeta, abstractmethod
import time
import numpy as np
import random
from collections import Counter
from tqdm import tqdm


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_name, learning_rate=1e-3, network='MLP', batch_size=32,
                 device='cuda', epochs=1000, prt_interval=200, val_split=0.8, random_state=42):
        '''
        Abstract base class for all models.

        Parameters:
        -----------
        model_name : str
            Name of the model.
        learning_rate : float
            Learning rate for the optimizer.
        network : str
            Name of the network.
        batch_size : int
            Batch size for training.
        device : str
            Device to train the model on. Either 'cpu' or 'cuda'.
        epochs : int
            Number of epochs to train the model.
        prt_interval: int
            interval at which loss should be printed 
        random_state : int
            Random seed for reproducibility.
        '''

        self.model_name = model_name
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.random_state = random_state
        self.prt_interval = prt_interval
        self.val_split = val_split

        self.set_seed(self.random_state)
        return

    def fit(self, X, y):
        '''
        
        '''
        print("Starting training...")
        #print(f"Using {self.device} for training")

        self.train_data = X
        self.train_label = y
        
        self.train_loader, self.val_loader, self.net, self.criterion = self.training_prepare(
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

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=0)

        self.net.train()
        for epoch in range(self.epochs):
            
            total_loss = 0.0
            for batch_x in self.train_loader:
                optimizer.zero_grad()
                loss = self.training_forward(batch_x, self.net, self.criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss/len(self.train_loader)
            
            total_loss = 0.0
            with torch.inference_mode():
                for batch_x in self.val_loader:
                    loss = self.training_forward(batch_x, self.net, self.criterion)
                    total_loss+=loss.item()
            
            val_loss = total_loss/len(self.val_loader)
            
            if epoch == 0 or ((epoch + 1) % self.prt_interval == 0) or (epoch == self.epochs - 1):
                print("Epoch: {:3d},  training loss: {:.6f}, validation loss: {:.6f} ".format(
                            epoch+1, train_loss, val_loss))
        return

    def _inference(self):
        
        self.net.eval()
        for epoch in range(self.epochs):

            with torch.inference_mode():

                for data in self.test_loader:
                    batch_x, batch_y = data, None
                    y_pred, _ = self.inference_forward(
                        batch_x[0], batch_y, self.net, self.criterion)

        print("Finished making predictions!")
        return y_pred

    def _validate(self):

        self.net.eval()
        for epoch in range(self.epochs):
            t1 = time.time()
            total_loss = 0

            with torch.inference_mode():

                for i, data in enumerate(tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch'), 0):
                    batch_x, batch_y = data
                    _, loss = self.inference_forward(
                        batch_x, batch_y, self.net, self.criterion)
                    total_loss += loss.item()

            val_loss = total_loss/len(self.test_loader)
            t = time.time()-t1
            print('Epoch: {:3d}, time:{:.1f} seconds'.format(
                epoch+1, t))
        print('Validation loss: {:.6f}'.format(val_loss))

        return val_loss

    def classification_report(self, y_true, y_pred):
        """
        Generate a classification report with precision, recall, f1-score, and support.
        
        Args:
            y_true (list): Ground truth (correct) target values.
            y_pred (list): Estimated targets as returned by a classifier.
            
        Returns:
            str: The classification report as a formatted string.
        """
        # Calculate the true positives, false positives, and false negatives for each label
        labels = set(y_true).union(set(y_pred))
        tp, fp, fn = {}, {}, {}
        for label in labels:
            tp[label] = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == label)
            fp[label] = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == label)
            fn[label] = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yt == label)
        
        # Calculate the support (number of instances) for each label
        support = Counter(y_true)
        
        # Calculate precision, recall, and f1-score for each label
        report = ""
        for label in labels:
            precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] != 0 else 0
            recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] != 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            
            report += f"Label: {label}\n"
            report += f"  Precision: {precision:.3f}\n"
            report += f"  Recall: {recall:.3f}\n"
            report += f"  F1-Score: {f1_score:.3f}\n"
            report += f"  Support: {support[label]}\n\n"
        
        return report
    
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
