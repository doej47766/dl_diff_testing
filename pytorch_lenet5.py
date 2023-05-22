"""PyTorch implementation of the Lenet5 model."""

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms

import datetime
import time

class PyTorchLenet5Mod(Module):
    """This class implements the Lenet5 model using PyTorch.

    Arguments
    ---------
    initializer: function
        The weight initialization function from the torch.nn.init module that is used to initialize
        the initial weights of the models.
    dropout: float
        The dropout rate that will be considered during training.
    """
    
    def __init__(self, initializer, dropout):
        super(PyTorchLenet5Mod, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)#, bias=True)
        initializer(self.conv1.weight)
        self.conv2 = nn.Conv2d(6, 16, 5)#, bias=True)
        initializer(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)#, bias=True)
        initializer(self.fc1.weight)
        self.fc2 = nn.Linear(120, 84)#, bias=True)
        initializer(self.fc2.weight)
        self.fc3 = nn.Linear(84, 10)#, bias=True)
        initializer(self.fc3.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_training=False):
        """This function implements the forward pass of the model.
        
        Arguments
        ---------
        x: Tensor
            The set of samples the model is to infer.
        is_training: boolean
            This indicates whether the forward pass is occuring during training
            (i.e., if we should consider dropout).
        """
        # pool size = 2
        # input size = (28, 28), output size = (14, 14), output channel = 6
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        # pool size = 2
        # input size = (10, 10), output size = (5, 5), output channel = 16
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        # flatten as one dimension
        # x = x.view(x.size()[0], -1)
        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        if is_training:
            x = self.dropout(x)

        # input dim = 16*5*5, output dim = 120
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)

        if is_training:
            x = self.dropout(x)

        # input dim = 120, output dim = 84
        x = torch.nn.functional.relu(x)
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x

    def train_pytorch(self, optimizer, epoch, train_loader, device, data_type, log_interval):
        """This function implements a single epoch of the training process of the PyTorch model.

        Arguments
        ---------
        self: PyTorchLenet5Mod
            The model that is to be trained.
        optimizer: torch.nn.optim
            The optimizer to be used during the training process.
        epoch: int
            The epoch associated with the training process.
        train_loader: DataLoader
            The DataLoader that is used to load the training data during the training process.
            Note that the DataLoader loads the data according to the batch size
            defined with it was initialized.
        device: string
            The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
        data_type: string
            This string indicates whether mixed precision is to be used or not.
        log_interval: int
            The interval at which the model logs the process of the training process
            in terms of number of batches passed through the model.
        """
        
        # State that you are training the model
        self.train()

        # define loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        if data_type == 'mixed':
            scaler = torch.cuda.amp.GradScaler()

        # Iterate over batches of data
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # if device == 'gpu':
            #     data, target = data.cuda(), target.cuda()

            # Wrap the input and target output in the `Variable` wrapper
            # data, target = Variable(data), Variable(target)

            # Clear the gradients, since PyTorch accumulates them
            optimizer.zero_grad()

            if data_type == 'mixed':
                with torch.cuda.amp.autocast():
                    # Forward propagation
                    output = self(data, is_training=True)

                    loss = loss_fn(output, target)
                    # print(loss.device)

                # Backward propagation
                scaler.scale(loss).backward()

                # Update the parameters(weight,bias)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward propagation
                output = self(data, is_training=True)

                loss = loss_fn(output, target)
                # print(loss.device)

                # Backward propagation
                loss.backward()

                # Update the parameters(weight,bias)
                optimizer.step()

            if log_interval == -1:
                continue

            # print log
            if batch_idx % log_interval == 0:
                print('Train set, Epoch {}\tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        # return model

    def test_pytorch(self, test_loader, device, data_type):
        """This function implements the testing process of the PyTorch model and returns the accuracy
        obtained on the testing dataset.

        Arguments
        ---------
        model: torch.nn.Module
            The model that is to be tested.
        test_loader: DataLoader
            The DataLoader that is used to load the testing data during the testing process.
            Note that the DataLoader loads the data according to the batch size
            defined with it was initialized.
        device: string
            The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
        data_type: string
            This string indicates whether mixed precision is to be used or not.

        """
        
        
        # State that you are testing the model; this prevents layers e.g. Dropout to take effect
        self.eval()

        with torch.no_grad():

            # Init loss & correct prediction accumulators
            test_loss = 0
            correct = 0

            # define loss function
            loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
            # Iterate over data
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # data, target = Variable(data), Variable(target)
                if data_type == 'mixed':
                    with torch.cuda.amp.autocast():
                        # Forward propagation
                        output = self(data).detach()

                        # Calculate & accumulate loss
                        test_loss += loss_fn(output, target).detach()

                        # Get the index of the max log-probability (the predicted output label)
                        # pred = np.argmax(output.data, axis=1)
                        pred = output.data.argmax(dim=1)

                        # If correct, increment correct prediction accumulator
                        correct = correct + (pred == target.data).sum()
                else:
                    # Forward propagation
                    output = self(data).detach()

                    # Calculate & accumulate loss
                    test_loss += loss_fn(output, target).detach()

                    # Get the index of the max log-probability (the predicted output label)
                    # pred = np.argmax(output.data, axis=1)
                    pred = output.data.argmax(dim=1)

                    # If correct, increment correct prediction accumulator
                    correct = correct + (pred == target.data).sum()

            # Print log
            test_loss /= len(test_loader.dataset)
            print('\nTest set, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            return 100. * correct / len(test_loader.dataset)
        
        
def generate_pytorch_dataloader(X_train_padded, X_test_padded, X_test_padded_ext, y_train, y_test, y_test_ext, batch_size, device):
    """This functions generate the dataset loaders for PyTorch. The function returns the
    training dataloader (i.e., pytorch_train_loader), the testing dataloader (i.e.,
    pytorch_test_loader) and the larger testing dataset dataloader (i.e.,
    pytorch_test_loader_ext) used during the inference phase.  
    
    Arguments
    ---------
    X_train_padded: numpy array
        Padded training dataset.
    X_test_padded: numpy array
        Padded testing dataset.
    X_test_padded_ext: numpy array
        Padded larger testing dataset.
    y_train: numpy array
        Labels of the training set.
    y_test: numpy array
        Labels of the testing set.
    y_test_ext: numpy array
        Labels of the larger testing dataset.
    batch_size: int
        The batch size that will be used for training
        and testing the model.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    """
    
    
    X_torch_train_mnist = torch.from_numpy(X_train_padded).view(X_train_padded.shape[0], 1, 32, 32).type(torch.float32).to(device)
    X_torch_test_mnist = torch.from_numpy(X_test_padded).view(X_test_padded.shape[0], 1, 32, 32).type(torch.float32).to(device)
    X_torch_test_mnist_ext = torch.from_numpy(X_test_padded_ext).view(X_test_padded_ext.shape[0], 1, 32, 32).type(torch.float32).to(device)
            
    y_torch_train_mnist = torch.LongTensor(y_train).to(device)
    y_torch_test_mnist = torch.LongTensor(y_test).to(device)
    y_torch_test_mnist_ext = torch.LongTensor(y_test_ext).to(device)
    

    pytorch_lenet5_train_dataset = TensorDataset(X_torch_train_mnist, y_torch_train_mnist)
    pytorch_lenet5_test_dataset = TensorDataset(X_torch_test_mnist, y_torch_test_mnist)
    pytorch_lenet5_test_dataset_ext = TensorDataset(X_torch_test_mnist_ext, y_torch_test_mnist_ext)

    pytorch_train_loader = DataLoader(pytorch_lenet5_train_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader = DataLoader(pytorch_lenet5_test_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader_ext = DataLoader(pytorch_lenet5_test_dataset_ext, batch_size=batch_size, shuffle=False)

    return pytorch_train_loader, pytorch_test_loader, pytorch_test_loader_ext


def pytorch_training_phase(model, optimizer, train_loader, test_loader, n_epochs, device, data_type, experiment):
    """"This function mplements the training phase of the PyTorch implementation of the LSTM model
    and returns the training time, the training timestamps (corresponding to when the training
    process began and when it ended) and the accuracy obtained on the testing dataset. The function
    also saves the model. 
    
    Arguments
    ---------
    model: torch.nn.Module
        The model that is to be trained.
    optimizer: torch.nn.optim
        The optimizer to be used during the training process.
    train_loader: DataLoader
        The DataLoader that is used to load the testing data during the testing process.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    test_loader: DataLoader
        The DataLoader that is used to load the testing data during the testing process.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    n_epochs: int
        The number of epochs for the training process.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    experiment: string
        The string that is used to identify the model (i.e., the set of configurations the model uses).
    
    """
    train_start_timestamp = datetime.datetime.now()
    start = time.time()

    for epoch in range(1, n_epochs+1):
        model.train_pytorch(optimizer, epoch, train_loader, device, data_type, log_interval=60000)

    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()

    
    start = time.time()
    accuracy = model.test_pytorch(test_loader, device, data_type)
    inference_time = (time.time() - start)

    #Save the model
    torch.save(model.state_dict(), './models/lenet5/{}/model'.format(experiment))

    return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp

def pytorch_inference_phase(model, experiment, pytorch_test_loader_ext, device, data_type):
    """This function implements the inference phase of the PyTorch implementation of the LSTM model.
    The function returns the inference timestamps (corresponding to when the inference began and when
    it ended). 
    
    Arguments
    ---------
    model: torch.nn.Module
        The model that is to be evaluated (the model acts as a placeholder into which the weights of
        the trained model will be loaded).
    experiment: string
        The string that is used to identify the model (i.e., the set of configurations the model uses).
    pytorch_test_loader_ext: DataLoader
        The DataLoader that is used to load the larger testing data during the inference phase.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    """
    model.load_state_dict(torch.load('./models/lenet5/{}/model'.format(experiment)))
    model.eval()

    inference_start_timestamp = datetime.datetime.now()
    accuracy = model.test_pytorch(pytorch_test_loader_ext, device, data_type)
    inference_end_timestamp = datetime.datetime.now()
    print('Accuracy: {}'.format(accuracy))
    
    return inference_start_timestamp, inference_end_timestamp
