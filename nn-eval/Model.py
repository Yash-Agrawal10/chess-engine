import torch
import torch.nn as nn

class EvalNN(nn.Module):

    def __init__(self):
        super(EvalNN, self).__init__()
        inputs = 81920 #change this to the number of features
        self.layer_1 = nn.Linear(inputs, 512)
        self.layer_2 = nn.Linear(512, 32)
        self.layer_3 = nn.Linear(32, 32)
        self.layer_4 = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)

        self.activation = nn.ReLU()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X):
        X = self.layer_1(X)
        X = self.activation(X)
        X = self.layer_2(X)
        X = self.activation(X)
        X = self.layer_3(X)
        X = self.activation(X)
        X = self.layer_4(X)
        return X
    
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (inputs, targets) in dataloader:
                self.optimizer.zero_grad()
                yhat = self.forward(inputs)
                loss = self.criterion(yhat, targets)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(epoch, total_loss / len(dataloader))

    def evaluate(self, dataloader):
        total_loss = 0
        for (inputs, targets) in dataloader:
            yhat = self(inputs)
            total_loss += self.criterion(yhat, targets).item()
        print(total_loss / len(dataloader))