from torch.utils.data import Dataset
import sqlite3
import pandas as pd
import process

class EvalDataset(Dataset):

    def __init__(self, sqlpath, count, offset):
        con = sqlite3.connect(sqlpath)
        query = f'SELECT fen, eval FROM evals LIMIT {count} OFFSET {offset}'
        df = pd.read_sql(query, con)
        df = df.apply(process.process_row, axis=1)
        self.X = df["pieceboards"]
        self.Y = df["normalized_eval"]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        input = self.X[idx]
        output = self.Y[idx]
        return input, output