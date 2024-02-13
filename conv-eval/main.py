from Dataset import EvalDataset
from Model import EvalNN
import torch
from torch.utils.data import DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')
import os
import process

# Create pickle file if not found
path = "data/chessData.pkl"
if not os.path.exists(path):
    print("csv to pickle not found")
    print("converting csv to pickle")
    csv_path = "data/chessData.csv"
    pickle_path = "data/chessData.pkl"
    count = 1000000
    process.csv_to_pickle(csv_path, pickle_path, count)
    print("csv to pickle complete")

# Initialize Dataset
k = 5   # must divide 1 evenly as a decimal
full_dataset = EvalDataset(path)
datasets = random_split(full_dataset, [1/k] * k)
print("Initialized Datasets")

# Initialize DataLoaders
train_dls = []
for dataset in datasets:
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    train_dls.append(dataloader)
test_dl = train_dls.pop()
print("Initialized DataLoaders")

# Define / Load Model
model = EvalNN()
# model = torch.load("eval-nn.pt")
print("Initialized Model")

# Train Model
for i in range(len(train_dls)):
    for j in range(len(train_dls)):
        if i != j:
            print(f'Training on {j}, Evaluating on {i}')
            model.train(train_dls[j], 30)
    model.evaluate(train_dls[i])

# Evaluate Model
model.evaluate(test_dl)

# Save Model
# torch.save(model, "eval-nn.pt")