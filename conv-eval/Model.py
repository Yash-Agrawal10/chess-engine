import torch
import torch.nn as nn

class EvalNN(nn.Module):

    def __init__(self):
        super(EvalNN, self).__init__()
        inputs = 768
        

        self.activation = nn.ReLU()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, mine, theirs):
        pass
    
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