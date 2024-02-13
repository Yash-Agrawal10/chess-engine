import torch
import torch.nn as nn

class EvalNN(nn.Module):

    def __init__(self):
        super(EvalNN, self).__init__()
        inputs = 768
        self.dense1 = nn.Linear(inputs, 1048)
        self.dense2 = nn.Linear(1048, 500)
        self.dense3 = nn.Linear(500, 50)
        self.dense4 = nn.Linear(50, 1)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.dense3.weight)
        nn.init.xavier_uniform_(self.dense4.weight)

        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X):
        X = self.activation(self.dense1(X))
        X = self.activation(self.dense2(X))
        X = self.activation(self.dense3(X))
        X = self.out_activation(self.dense4(X))
        return X
    
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (input, target) in dataloader:
                self.optimizer.zero_grad()
                yhat = self.forward(input)
                loss = self.criterion(yhat, target)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print("Epoch: ", epoch, "Loss: ", total_loss / len(dataloader))

    def evaluate(self, dataloader):
        total_loss = 0
        for (input, target) in dataloader:
            yhat = self.forward(input)
            total_loss += self.criterion(yhat, target).item()
        average_loss = total_loss / len(dataloader)
        print("Average Loss: ", average_loss)