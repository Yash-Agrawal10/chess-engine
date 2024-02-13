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
        self.mine = df["mine"]
        self.theirs = df["theirs"]
        assert(self.mine.size == self.theirs.size)
        self.Y = df["normalized_eval"]

    def __len__(self):
        return self.mine.size
    
    def __getitem__(self, idx):
        mine = self.mine.at[idx]
        theirs = self.theirs.at[idx]
        eval = self.Y.at[idx]
        return [mine, theirs, eval]