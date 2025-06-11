# src/data_download.py
import pandas as pd
from datasets import load_dataset

def main():
    # Load the dataset from Huggingface
    print("Loading LIAR2 dataset from Huggingface...")
    ds = load_dataset("chengxuphd/liar2")
    
    df = pd.concat([ds[s].to_pandas() for s in ['train', 'validation', 'test']], ignore_index=True)
    print("Loaded:", df.shape, "rows")
    df.to_csv("data/liar2_full.csv", index=False)
    print("Saved combined dataset to data/liar2_full.csv")

if __name__ == "__main__":
    main()
