import pandas as pd

df = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/023_train.pkl')
print(df.shape)
print(df.head())