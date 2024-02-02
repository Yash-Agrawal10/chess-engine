from Dataset import EvalDataset
from Model import EvalNN

import torch
from torch.utils.data import DataLoader, random_split

# import numpy as np
# import matplotlib.pyplot as plt

# Initialize Dataset
sqlpath = "/Users/User/sqlite/chess-evals.db"
lower, upper = 1, 100000
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
# model.load_state_dict(torch.load("./eval-nn.pt"))
print("Initialized Model")

# Train Model
for dataloader in train_dls:
    model.train(dataloader, 30)

# Evaluate Model
model.evaluate(test_dl)

# Save Model
torch.save(model.state_dict(), "eval-nn.pt")

# Test Model on Individual Inputs
import sqlite3
import pandas as pd
import processing
con = sqlite3.connect(sqlpath)
query = "select fen, eval from evaluations limit 100 offset 10000000;"
df = pd.read_sql(query, con)
df = df.apply(processing.process_row, axis=1)
for index in df.index:
    fen = df.at[index, "fen"]
    stockfish_eval = df.at[index, "eval"]
    halfkp = df.at[index, "halfkp"]
    white_to_move = df.at[index, "white_to_move"]
    engine_eval = model(torch.tensor(halfkp)).item()
    engine_eval = processing.denormalize_eval(engine_eval, white_to_move)
    print(stockfish_eval, engine_eval)