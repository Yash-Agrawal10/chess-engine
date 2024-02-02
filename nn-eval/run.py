from Dataset import EvalDataset
from Model import EvalNN
import torch

import sqlite3
import pandas as pd
import processing

# Define and Load Model
model = EvalNN()
model.load_state_dict(torch.load("./eval-nn.pt"))
print("Initialized Model")

# Initialize Dataset
sqlpath = "/Users/User/sqlite/chess-evals.db"
con = sqlite3.connect(sqlpath)
query = "select fen, eval from evaluations limit 10 offset 10000000;"
df = pd.read_sql(query, con)
df = df.apply(processing.process_row, axis=1)

# Run Model on Datapoints
for index in df.index:
    fen = df.at[index, "fen"]
    stockfish_eval = df.at[index, "eval"]
    mine = torch.tensor([df.at[index, "mine"]])
    theirs = torch.tensor([df.at[index, "theirs"]])
    white_to_move = df.at[index, "white_to_move"]
    engine_output = model.forward(mine, theirs).item()
    engine_eval = processing.denormalize_eval(engine_output, white_to_move)
    print(stockfish_eval, engine_eval)