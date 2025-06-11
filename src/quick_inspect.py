# quick_inspect.py
import pandas as pd

df = pd.read_csv("data/liar2_full.csv", nrows=0)
print(df.columns.tolist())
