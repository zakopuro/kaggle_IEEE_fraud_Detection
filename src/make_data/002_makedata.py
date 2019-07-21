# アンダーサンプリングしてみる

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


def under_sampling(df):
    print('under_sampling')
    df_data0 = df[df['isFraud'] == 0]
    df_data1 = df[df['isFraud'] == 1]
    data1_size = len(df_data1)
    data0_index = df_data0.index
    random_index = np.random.choice(data0_index,data1_size,replace=False)
    df_choice_data0 = df.loc[random_index]
    df = pd.concat([df_data1,df_choice_data0],ignore_index=True)

    return df


def main():
    df_train_iden = pd.read_csv('../IEEE_Fraud_Detection/input/train_identity.csv')
    df_train_trans = pd.read_csv('../IEEE_Fraud_Detection/input/train_transaction.csv')
    df_test_iden = pd.read_csv('../IEEE_Fraud_Detection/input/test_identity.csv')
    df_test_trans = pd.read_csv('../IEEE_Fraud_Detection/input/test_transaction.csv')

    df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
    df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')
    print('test1')
    # ------- 欠損値処理 ----------
    df_train = null_data_processing(df_train)
    df_test = null_data_processing(df_test)
    # ------- 不要カラム削除 -------
    df_train = del_colmuns(df_train)
    df_test = del_colmuns(df_test)
    # ------- ラベルエンコーダー -----
    df_train,df_test = label_encoder(df_train,df_test)

    # -------- アンダーサンプリング ---------
    df_train = under_sampling(df_train)

    print(df_train['isFraud'].value_counts())

    df_train.to_csv('../IEEE_Fraud_Detection/src/make_data/data/002_train.csv',index=False)
    df_test.to_csv('../IEEE_Fraud_Detection/src/make_data/data/002_test.csv',index=False)



if __name__ == "__main__":
    main()