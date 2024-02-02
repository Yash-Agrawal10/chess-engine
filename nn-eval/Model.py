import torch
import torch.nn as nn

class EvalNN(nn.Module):

    def __init__(self):
        super(EvalNN, self).__init__()
        inputs = 40960 # there are 2 inputs of this size
        self.condense_mine = nn.Linear(inputs, 256)
        self.condense_theirs = nn.Linear(inputs, 256)
        self.connected_1 = nn.Linear(512, 32)
        self.connected_2 = nn.Linear(32, 32)
        self.connected_3 = nn.Linear(32, 1)

        nn.init.xavier_uniform_(self.condense_mine.weight)
        nn.init.xavier_uniform_(self.condense_theirs.weight)
        nn.init.xavier_uniform_(self.connected_1.weight)
        nn.init.xavier_uniform_(self.connected_2.weight)
        nn.init.xavier_uniform_(self.connected_3.weight)

        self.activation = nn.ReLU()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, mine, theirs):
        mine = self.condense_mine(mine)
        theirs = self.condense_theirs(theirs)
        X = torch.cat((mine, theirs), 1)
        X = self.activation(X)
        X = self.connected_1(X)
        X = self.activation(X)
        X = self.connected_2(X)
        X = self.activation(X)
        X = self.connected_3(X)
        return X
    
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (mine, theirs, targets) in dataloader:
                self.optimizer.zero_grad()
                yhat = self.forward(mine, theirs)
                loss = self.criterion(yhat, targets)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(epoch, total_loss / len(dataloader))

    def evaluate(self, dataloader):
        total_loss = 0
        for (mine, theirs, targets) in dataloader:
            yhat = self(mine, theirs)
            total_loss += self.criterion(yhat, targets).item()
        print(total_loss / len(dataloader))