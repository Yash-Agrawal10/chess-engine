from Dataset import EvalDataset
from Model import EvalNN

import torch
from torch.utils.data import DataLoader, random_split

# Initialize Dataset
sqlpath = "/Users/User/sqlite/chess-evals.db"
lower, upper = 500000, 600000
k = 10   # must divide 1 evenly as a decimal
full_dataset = EvalDataset(sqlpath, lower, upper)
datasets = random_split(full_dataset, [1/k] * k)
print("Initialized Datasets")

# Initialize DataLoaders
train_dls = []
for dataset in datasets:
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    train_dls.append(dataloader)
test_dl = train_dls.pop()
print("Initialized DataLoaders")

# Define and Load Model
model = EvalNN()
model.load_state_dict(torch.load("./eval-nn.pt"))
print("Initialized Model")

# Train Model
for dataloader in train_dls:
    model.train(dataloader, 30)

# Evaluate Model
model.evaluate(test_dl)

# Save Model
torch.save(model.state_dict(), "eval-nn.pt")