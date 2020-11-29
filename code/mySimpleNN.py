import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self, D_in, H, D_out):
        super(mySimpleNN, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H)
        # self.linear3 = torch.nn.Linear(H, H)
        # self.linear4 = torch.nn.Linear(H, H)
        # self.linear5 = torch.nn.Linear(H, D_out)

        self.linear1 = torch.nn.Linear(D_in, H)
        #self.bn1 = torch.nn.BatchNorm1d(num_features=H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H,H)
        self.linear4 = torch.nn.Linear(H,H)
        self.linear5 = torch.nn.Linear(H, D_out)

        self.dropout = torch.nn.Dropout(p=0.1)
    # TODO: Define a simple neural network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        # h_relu = relu(self.linear1(x))
        # linear2 = relu(self.linear2(h_relu))
        # linear2 = self.dropout(linear2)
        # linear3 = relu(self.linear3(linear2))
        # linear4 = relu(self.linear4(linear3))
        # y_hat = sigmoid(self.linear5(linear4))
        #print(min(y_hat))

        x = relu((self.linear1(x)))

        #x = self.dropout(x)
        x = relu(self.linear2(x))
        #batchnorm was the problem...
        #x = (self.bn1(self.linear3(x)))
        x = self.linear3(x)
        x = relu(self.linear4(x))
        y_hat = sigmoid(self.linear5(x))

    # TODO: Define the network forward propagation from x -> y_hat

        return y_hat

