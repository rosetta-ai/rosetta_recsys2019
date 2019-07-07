'''
Transform files from csv to pickle for faster reading
'''
import pickle
import pandas as pd

df = pd.read_csv('../input/train.csv')
with open('../input/train_v2.p','wb') as f:
    pickle.dump(df, f)
    
df = pd.read_csv('../input/test.csv')
with open('../input/test_v2.p','wb') as f:
    pickle.dump(df, f)
    
df = pd.read_csv('../input/item_metadata.csv')
with open('../input/item_metadata.p','wb') as f:
    pickle.dump(df, f)    