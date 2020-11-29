import torch
from torchsummary import summary
from myNetworkTrainer import train_and_validate, writeHistoryPlots

from myImageNN import MyCNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: ", device)

    # TODO: train and test a CNN
    print('##########################')
    print('Testing Convolutional Neural Net')
    cnnModel = MyCNN()


    from prettytable import PrettyTable


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


    count_parameters(cnnModel)

    criterion = torch.nn.BCELoss()  # Cost function - torch.nn.XXX loss functions
    #optimizer = torch.optim.SGD(cnnModel.parameters(), lr=0.1)  # Optimizer algorithm - torch.optim.XXX function
    optimizer = torch.optim.Adam(cnnModel.parameters(), lr=0.001)
    # TODO: Your might also want to change the batchSize and number of epochs depending on your optimizer configuration
    finalCNNmodel, cnnHistory = train_and_validate(cnnModel, device, criterion, optimizer, epochs=200, batchSize=512)
    writeHistoryPlots(cnnHistory, 'cnnModel', 'output/')
