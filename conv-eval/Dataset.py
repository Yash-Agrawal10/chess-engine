from torch.utils.data import Dataset
import pandas as pd

class EvalDataset(Dataset):

    def __init__(self, path):
        #df = pd.read_csv(path, nrows=count)
        #df = df.apply(process.process_row, axis=1)
        #df = df.dropna()
        df = pd.read_pickle(path)
        if df.isnull().values.any():
            raise Exception("Null values found in dataset")
        self.X = df["pieceboards"]
        self.Y = df["eval"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        input = self.X[idx]
        output = self.Y[idx]
        return [input, output]