import torch
from torchsummary import summary
from myNetworkTrainer import train_and_validate, writeHistoryPlots

from myImageNN import MyLogRegNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: ", device)

    # TODO: train and test a logistic regression classifier implemented as a neural network
    print('##########################')
    print('Testing Logistic Regression')
    logRegModel = MyLogRegNN(input_dim= 32*32*3, output_dim=1)

    from prettytable import PrettyTable

    # see https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    count_parameters(logRegModel)

    criterion = torch.nn.BCELoss()  # Cost function - torch.nn.XXX loss functions
    optimizer = torch.optim.SGD(logRegModel.parameters(), lr=0.01)  # Optimizer algorithm - torch.optim.XXX function
    # TODO: Your might also want to change the batchSize and number of epochs depending on your optimizer configuration
    finallogRegmodel, logRegHistory = train_and_validate(logRegModel, device, criterion, optimizer, epochs=200, batchSize=500)
    writeHistoryPlots(logRegHistory, 'logRegModel', 'output/')
