import pandas as pd

import newspaper
from newspaper import Article
import os 
from multiprocessing import Pool

df = None
for i in range(1, 43):
    dfi = pd.read_csv(f'dataset_{i}.csv')
    if df is None:
        df = dfi
    else:
        df = pd.concat([df,dfi], ignore_index=True)

df.loc[:, ~df.columns.str.match('Unnamed')].to_csv('dataset.csv')