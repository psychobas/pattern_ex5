import torch
import torch.nn.functional as F


class MyLogRegNN(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MyLogRegNN, self).__init__()
        # TODO: Define a logistic regression classifier as a neural network
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        sigmoid = torch.nn.Sigmoid()
        #x = x.reshape(-1, 32*32)
        y_hat = self.linear(x.view(-1, 32*32*3))
        y_hat = sigmoid(y_hat)
        return y_hat


class MyFullyConnectedNN(torch.nn.Module):
    def __init__(self, input_dim, H, output_dim):
        super(MyFullyConnectedNN, self).__init__()
        # TODO: Define a fully connected neural network
        self.linear1 = torch.nn.Linear(input_dim, H)
        # self.bn1 = torch.nn.BatchNorm1d(num_features=H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, output_dim)

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()

        x = relu((self.linear1(x.view(-1, 32*32*3))))
        #x = self.dropout(x)
        x = relu(self.linear2(x))
        #batchnorm was the problem...
        #x = (self.bn1(self.linear3(x)))
        x = relu(self.linear3(x))
        y_hat = sigmoid(self.linear4(x))

        return y_hat


class MyCNN(torch.nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # TODO: Define a convolutional neural network
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        y_hat = torch.sigmoid(y_hat)
        return y_hat


