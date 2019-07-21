import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

def null_data_processing(df):
    df = df.fillna(-9999)
    return df


def label_encoder(df_train,df_test):
    for col in tqdm(df_test.columns):
        if df_train[col].dtype == 'object' or df_test[col].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_train[col].values) + list(df_test[col].values))
            df_train[col] = lbl.transform(list(df_train[col].values))
            df_test[col] = lbl.transform(list(df_test[col].values))
    return df_train,df_test


def del_colmuns(df):

    return df


def main():
    df_train_iden = pd.read_csv('../IEEE_Fraud_Detection/input/train_identity.csv')
    df_train_trans = pd.read_csv('../IEEE_Fraud_Detection/input/train_transaction.csv')
    df_test_iden = pd.read_csv('../IEEE_Fraud_Detection/input/test_identity.csv')
    df_test_trans = pd.read_csv('../IEEE_Fraud_Detection/input/test_transaction.csv')

    df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
    df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')

    # ------- 欠損値処理 ----------
    df_train = null_data_processing(df_train)
    df_test = null_data_processing(df_test)
    # ------- 不要カラム削除 -------
    df_train = del_colmuns(df_train)
    df_test = del_colmuns(df_test)
    # ------- ラベルエンコーダー -----
    df_train,df_test = label_encoder(df_train,df_test)

    df_train.to_csv('../IEEE_Fraud_Detection/src/make_data/data/001_train.csv')
    df_test.to_csv('../IEEE_Fraud_Detection/src/make_data/data/001_test.csv')



if __name__ == "__main__":
    main()