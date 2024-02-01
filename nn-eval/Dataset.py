from torch.utils.data import Dataset
import sqlite3
import numpy as np
import pandas as pd
import processing

class EvalDataset(Dataset):

    def __init__(self, sqlpath, lower, upper):
        con = sqlite3.connect(sqlpath)
        query = "select fen, eval from evaluations where id between " + str(lower) + " and " + str(upper) + ";"
        df = pd.read_sql(query, con)
        df = df.apply(processing.process_row, axis=1)
        self.X = df["halfkp"]
        self.Y = df["normalized_eval"]

    def __len__(self):
        return self.X.size
    
    def __getitem__(self, idx):
        halfkp = self.X.at[idx]
        eval = self.Y.at[idx]
        return [halfkp, eval]