

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import datetime



def make_data_feature(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    # df_train['year'] = df_train['TransactionDT'].dt.year
    df['month'] = df['TransactionDT'].dt.month
    df['week'] = df['TransactionDT'].dt.dayofweek
    df['hour'] = df['TransactionDT'].dt.hour
    df['day'] = df['TransactionDT'].dt.day

    return df


def chg_email_feature(df,email_list):
    email_yahoo = ['yahoo.co.jp','yahoo.co.uk','yahoo.com','yahoo.com.mx','yahoo.de','yahoo.es','yahoo.fr','ymail.com','frontier.com','frontiernet.net','rocketmail.com']
    email_microsoft = ['hotmail.co.uk','hotmail.com','hotmail.de','hotmail.es','hotmail.fr','outlook.com','outlook.es','live.com','live.com.mx','live.fr','msn.com']
    email_apple = ['icloud.com','mac.com''me.com']
    email_att = ['arodigy.net.mx','att.net','sbcglobal.net']
    email_cent = ['centurylink.net','embarqmail.com','q.com']
    email_aol = ['aim.com','aol.com']
    email_spec = ['twc.com','charter.net']
    email_google = [ 'gmail','gmail.com']
    email_choice_list = email_yahoo + email_microsoft + email_apple + email_att + email_cent + email_aol + email_spec + email_google
    email_other = [e for e in email_list if e not in email_choice_list]

    df.loc[df['P_emaildomain'].isin(email_yahoo),'P_emaildomain'] = 'Yahoo'
    df.loc[df['P_emaildomain'].isin(email_microsoft),'P_emaildomain'] = 'Microsoft'
    df.loc[df['P_emaildomain'].isin(email_apple),'P_emaildomain'] = 'Apple'
    df.loc[df['P_emaildomain'].isin(email_att),'P_emaildomain'] = 'ATT'
    df.loc[df['P_emaildomain'].isin(email_cent),'P_emaildomain'] = 'Cent'
    df.loc[df['P_emaildomain'].isin(email_aol),'P_emaildomain'] = 'AOL'
    df.loc[df['P_emaildomain'].isin(email_spec),'P_emaildomain'] = 'Spec'
    df.loc[df['P_emaildomain'].isin(email_google),'P_emaildomain'] = 'Google'
    df.loc[df['P_emaildomain'].isin(email_other),'P_emaildomain'] = 'Other'

    df.loc[df['R_emaildomain'].isin(email_yahoo),'R_emaildomain'] = 'Yahoo'
    df.loc[df['R_emaildomain'].isin(email_microsoft),'R_emaildomain'] = 'Microsoft'
    df.loc[df['R_emaildomain'].isin(email_apple),'R_emaildomain'] = 'Apple'
    df.loc[df['R_emaildomain'].isin(email_att),'R_emaildomain'] = 'ATT'
    df.loc[df['R_emaildomain'].isin(email_cent),'R_emaildomain'] = 'Cent'
    df.loc[df['R_emaildomain'].isin(email_aol),'R_emaildomain'] = 'AOL'
    df.loc[df['R_emaildomain'].isin(email_spec),'R_emaildomain'] = 'Spec'
    df.loc[df['R_emaildomain'].isin(email_google),'R_emaildomain'] = 'Google'
    df.loc[df['R_emaildomain'].isin(email_other),'R_emaildomain'] = 'Other'

    return df


def make_email_feature(df_train,df_test):
    # protonmail.comはisFraudが１の可能性が高い
    df_train.loc[(df_train['R_emaildomain'] == 'protonmail.com') & (df_train['P_emaildomain'] == 'protonmail.com'),'both_pro_mail'] = 1
    df_test.loc[(df_test['R_emaildomain'] == 'protonmail.com') & (df_test['P_emaildomain'] == 'protonmail.com'),'both_pro_mail'] = 1
    df_train['both_pro_mail'] = df_train['both_pro_mail'].fillna(0)
    df_test['both_pro_mail'] = df_test['both_pro_mail'].fillna(0)

    df_train.loc[(df_train['R_emaildomain'] == 'protonmail.com') | (df_train['P_emaildomain'] == 'protonmail.com'),'pro_mail'] = 1
    df_test.loc[(df_test['R_emaildomain'] == 'protonmail.com') | (df_test['P_emaildomain'] == 'protonmail.com'),'pro_mail'] = 1
    df_train['pro_mail'] = df_train['pro_mail'].fillna(0)
    df_test['pro_mail'] = df_test['pro_mail'].fillna(0)


    # pemail_list = list(df_train['P_emaildomain'].drop_duplicates()) + list(df_test['P_emaildomain'].drop_duplicates())
    # remail_list = list(df_train['R_emaildomain'].drop_duplicates()) + list(df_test['R_emaildomain'].drop_duplicates())
    # email_list = set(pemail_list + remail_list)
    # df_train = chg_email_feature(df_train,email_list)
    # df_test = chg_email_feature(df_test,email_list)

    return df_train,df_test

def featrue_engineering(df_train,df_test):
    # 行毎のnullの数
    df_train['Missing_count'] = df_train.isna().sum(axis=1)
    df_test['Missing_count'] = df_test.isna().sum(axis=1)
    # 日付情報を追加
    df_train = make_data_feature(df_train)
    df_test = make_data_feature(df_test)
    # email系
    df_train,df_test = make_email_feature(df_train,df_test)

    return df_train,df_test



def null_data_processing(df):
    df = df.fillna(-999)
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
    # 他の特徴量と相関が高いやつ p>=0.9
    # list_corr = ['C8','C11','C12','C2','C7','C5','D12','D2','D6','D7','V101','V102','V103','V105','V106','V11','V113','V126','V127','V128','V13','V132','V133','V134','V137',\
    #             'V140','V143','V145','V147','V149','V150','V151','V152','V153','V154','V155','V156','V157','V158','V159','V16','V160','V163','V164','V167','V168','V177','V178',\
    #             'V179','V182','V18','V183','V190','V192','V193','V196','V197','V198','V199','V201','V202','V203','V204','V205','V206','V207','V212','V213','V217','V218','V219',\
    #             'V222','V225','V231','V219','V232','V235','V236','V237','V239','V243','V244','V245','V249','V251','V253','V254','V256','V257','V259','V263','V265','V266','V269',\
    #             'V271','V272','V273','V274','V275','V276','V277','V278','V279','V28','V280','V292','V293','V294','V295','V296','V297','V298','V299','V30','V301','V302','V304',\
    #             'V306','V307','V308','V309','V31','V311','V315','V316','V317','V318','V32','V321','V322','V323','V324','V325','V326','V327','V328','V329','V33','V330','V331',\
    #             'V332','V333','V334','V336','V339','V34','V36','V40','V42','V43','V45','V48','V49','V5','V50','V51','V52','V54','V57','V58','V59','V60','V63','V64','V68','V70',\
    #             'V71','V72','V73','V74','V76','V79','V80','V81','V84','V85','V88','V89','V90','V91','V92','V93','V94','V96','V97','TransactionDT']
    list_corr = ['V300', 'V309', 'V111', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305',
        'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301',
        'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117',
        'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120','TransactionDT']
    # list_del_col = ['TransactionDT']

    df = df.drop(list_corr,axis=1)
    return df

def one_hot(df):
    onehot_list = ["card4", "M1", "M2", "M3","M4", "M5", "M6", "M7", "M8", "M9", "ProductCD"]
    df = pd.get_dummies(df, columns=onehot_list, dummy_na=True)
    return df

def main():
    df_train_iden = pd.read_csv('../IEEE_Fraud_Detection/input/train_identity.csv',index_col='TransactionID')
    df_train_trans = pd.read_csv('../IEEE_Fraud_Detection/input/train_transaction.csv',index_col='TransactionID')
    df_test_iden = pd.read_csv('../IEEE_Fraud_Detection/input/test_identity.csv',index_col='TransactionID')
    df_test_trans = pd.read_csv('../IEEE_Fraud_Detection/input/test_transaction.csv',index_col='TransactionID')
    df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
    df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')
    print('データ読み込み')
    # ------- 特徴量生成 -------
    df_train,df_test = featrue_engineering(df_train,df_test)
    print('特徴量生成')

    # ------- 欠損値処理 ----------
    df_train = null_data_processing(df_train)
    df_test = null_data_processing(df_test)
    print('欠損値')
    # ------- 不要カラム削除 -------
    df_train = del_colmuns(df_train)
    df_test = del_colmuns(df_test)
    print('カラム削除')
    # ------- OneHot ----------
    df_train = one_hot(df_train)
    df_test = one_hot(df_test)
    print('onehot')
    # ------- ラベルエンコーダー -----
    df_train,df_test = label_encoder(df_train,df_test)
    print('Label Encoder')


    df_train.to_csv('../IEEE_Fraud_Detection/src/make_data/data/007_train.csv')
    df_test.to_csv('../IEEE_Fraud_Detection/src/make_data/data/007_test.csv')



if __name__ == "__main__":
    main()