from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch

class AutoTrainer:
    def __init__(self, train_data, train_labels, test_data, test_labels, criterion, optimizer, num_epochs, lr=0.01, batch_size=32):
        self.train_dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size)
        self.test_dataset = TensorDataset(test_data, test_labels)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size)
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

    def train(self, model, device=None):
        optimizer = self.optimizer(model.parameters(), lr=self.lr)

        model.to(device)
        model.train()

        # TRAINING
        for epoch in tqdm(range(self.num_epochs)):
            num_obs = 0
            avg_train_loss = 0
            for batch_x, batch_y in self.train_loader:

                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                num_obs += len(batch_y)
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                avg_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = avg_train_loss/num_obs

        # EVALUATION
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            num_obs = 0
            avg_test_loss = 0

            for batch_x, batch_y in self.test_loader:

                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                # Forward pass
                outputs = model(batch_x)

                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                avg_test_loss += loss.item()

                # Calculate accuracy
                predicted = torch.round(outputs)
                correct_predictions += (predicted == batch_y).sum().item()
                num_obs += batch_y.size(0)

        # Calculate average loss and accuracy
        avg_test_loss = avg_test_loss / num_obs
        test_accuracy = correct_predictions / num_obs

        return avg_train_loss, avg_test_loss, test_accuracy



class AutoTrainerV2:
    def __init__(self, train_dataset, test_dataset, criterion, optimizer, num_epochs, lr=0.01, batch_size=32):
        self.train_loader=train_dataset
        if not isinstance(train_dataset, DataLoader):
            self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        self.test_loader=test_dataset
        if not isinstance(test_dataset, DataLoader):
            self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

    def train(self, model, device=None):

        optimizer = self.optimizer(model.parameters(), lr=self.lr)

        # TRAINING
        model.to(device)
        model.train()
        for epoch in tqdm(range(self.num_epochs)):
            num_obs = 0
            avg_train_loss = 0
            for batch_x, batch_y in self.train_loader:
                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                num_obs += len(batch_y)
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                avg_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = avg_train_loss/num_obs

        # EVALUATION
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            num_obs = 0
            avg_test_loss = 0

            for batch_x, batch_y in self.test_loader:
                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                # Forward pass
                outputs = model(batch_x)

                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                avg_test_loss += loss.item()

                # Calculate accuracy
                predicted = torch.round(outputs)
                correct_predictions += (predicted == batch_y).sum().item()
                num_obs += batch_y.size(0)

        # Calculate average loss and accuracy
        avg_test_loss = avg_test_loss / num_obs
        test_accuracy = correct_predictions / num_obs

        return avg_train_loss, avg_test_loss, test_accuracy