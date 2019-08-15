# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime
import itertools


def corret_card_id(x):
    x=x.replace('.0','')
    x=x.replace('-999','nan')
    return x

def define_indexes(df):
    
    # create date column
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    
    # df['year'] = df['TransactionDT'].dt.year
    df['month'] = df['TransactionDT'].dt.month
    df['dow'] = df['TransactionDT'].dt.dayofweek
    df['hour'] = df['TransactionDT'].dt.hour
    df['day'] = df['TransactionDT'].dt.day
   
    # create card ID 
    cards_cols= ['card1', 'card2', 'card3', 'card5']
    for card in cards_cols: 
        if '1' in card: 
            df['card_id']= df[card].map(str)
        else : 
            df['card_id']+= ' '+df[card].map(str)
    
    # small correction of the Card_ID
    df['card_id']=df['card_id'].apply(corret_card_id)

    return df


def make_featuter(df):
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')
    df['id_02_to_mean_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('mean')
    df['id_02_to_mean_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('mean')
    df['id_02_to_std_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('std')
    df['id_02_to_std_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('std')
    df['D15_to_mean_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('mean')
    df['D15_to_mean_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('mean')
    df['D15_to_std_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('std')
    df['D15_to_std_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('std')
    df['D15_to_mean_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('mean')
    df['D15_to_mean_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('mean')
    df['D15_to_std_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('std')
    df['D15_to_std_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('std')
    df['Missing_count'] = df.isna().sum(axis=1)
    df['TransactionAmt_Log'] = np.log(df['TransactionAmt'])
    df['card1/card2'] = np.log(df['card1']/df['card2'])
    df['card1/addr1'] = np.log(df['card1']/df['addr1'])
    df['card2/addr1'] = np.log(df['card2']/df['addr1'])

    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353
    df['uid'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)+'_'+df['card4'].astype(str)
    df['uid2'] = df['uid'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
    df['TransactionAmt_check'] = np.where(df['TransactionAmt'].isin(df['TransactionAmt']), 1, 0)
    df['TransactionAmt_to_mean_card3'] = df['TransactionAmt'] / df.groupby(['card3'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card5'] = df['TransactionAmt'] / df.groupby(['card5'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card3'] = df['TransactionAmt'] / df.groupby(['card3'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card5'] = df['TransactionAmt'] / df.groupby(['card5'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_mean_uid'] = df['TransactionAmt'] / df.groupby(['uid'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_uid2'] = df['TransactionAmt'] / df.groupby(['uid2'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_uid'] = df['TransactionAmt'] / df.groupby(['uid'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_uid2'] = df['TransactionAmt'] / df.groupby(['uid2'])['TransactionAmt'].transform('std')
    drop_list = ['uid','uid2']
    df.drop(drop_list,axis=1,inplace=True)

    # for imp_col in imp_cols:
    #     df['TransactionAmt_to_mean_' + imp_col] = df['TransactionAmt'] / df.groupby([imp_col])['TransactionAmt'].transform('mean')
    #     df['TransactionAmt_to_std_' + imp_col] = df['TransactionAmt'] / df.groupby([imp_col])['TransactionAmt'].transform('std')
    #     df['TransactionAmtlog_to_mean_' + imp_col] = df['TransactionAmt_Log'] / df.groupby([imp_col])['TransactionAmt_Log'].transform('mean')
    #     df['TransactionAmtlog_to_std_' + imp_col] = df['TransactionAmt_Log'] / df.groupby([imp_col])['TransactionAmt_Log'].transform('std')


    #https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_df-579654
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo',
              'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
              'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
              'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft',
              'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft',
              'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
              'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other',
              'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
              'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    # https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
    # df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    # df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df

def make_featuter2(train,test):
    card1_temp = pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False)
    train['card1_count'] = train['card1'].map(card1_temp)
    test['card1_count'] = test['card1'].map(card1_temp)
    card2_temp = pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False)
    train['card2_count'] = train['card2'].map(card2_temp)
    test['card2_count'] = test['card2'].map(card2_temp)
    # https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
    train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))
    test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))
    train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))
    test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))
    train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))
    test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))
    train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))
    test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))
    train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
    test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
    train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))
    test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

    # 自作
    # train['dist1_count_full'] = train['dist1'].map(pd.concat([train['dist1'], test['dist1']], ignore_index=True).value_counts(dropna=False))
    # test['dist1_count_full'] = test['dist1'].map(pd.concat([train['dist1'], test['dist1']], ignore_index=True).value_counts(dropna=False))
    # train['dist2_count_full'] = train['dist2'].map(pd.concat([train['dist2'], test['dist2']], ignore_index=True).value_counts(dropna=False))
    # test['dist2_count_full'] = test['dist2'].map(pd.concat([train['dist2'], test['dist2']], ignore_index=True).value_counts(dropna=False))
    # train['card1/card2_count_full'] = train['card1/card2'].map(pd.concat([train['card1/card2'], test['card1/card2']], ignore_index=True).value_counts(dropna=False))
    # test['card1/card2_count_full'] = test['card1/card2'].map(pd.concat([train['card1/card2'], test['card1/card2']], ignore_index=True).value_counts(dropna=False))
    # train['ccard1/addr1count_full'] = train['card1/addr1'].map(pd.concat([train['card1/addr1'], test['card1/addr1']], ignore_index=True).value_counts(dropna=False))
    # test['card1/addr1_count_full'] = test['card1/addr1'].map(pd.concat([train['card1/addr1'], test['card1/addr1']], ignore_index=True).value_counts(dropna=False))
    # train['card2/addr1_count_full'] = train['card2/addr1'].map(pd.concat([train['card2/addr1'], test['card2/addr1']], ignore_index=True).value_counts(dropna=False))
    # test['card2/addr1_count_full'] = test['card2/addr1'].map(pd.concat([train['card2/addr1'], test['card2/addr1']], ignore_index=True).value_counts(dropna=False))



    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

    # https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2',
                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)
        le = LabelEncoder()
        le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
        train[feature] = le.transform(list(train[feature].astype(str).values))
        test[feature] = le.transform(list(test[feature].astype(str).values))

    for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:
        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353
    i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain'
         ]
    for col in i_cols:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        train[col+'_fq_enc'] = train[col].map(fq_encode)
        test[col+'_fq_enc']  = test[col].map(fq_encode)
    for col in ['ProductCD','M4']:
        temp_dict = train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(
                                                            columns={'mean': col+'_target_mean'})
        temp_dict.index = temp_dict[col].values
        temp_dict = temp_dict[col+'_target_mean'].to_dict()
        train[col+'_target_mean'] = train[col].map(temp_dict)
        test[col+'_target_mean']  = test[col].map(temp_dict)



    return train,test


def one_val(train,test):
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

    many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
    many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))

    cols_to_drop.remove('isFraud')

    train.drop(cols_to_drop, axis=1, inplace=True)
    test.drop(cols_to_drop, axis=1, inplace=True)

    return train,test

def main():
    df_train = pd.read_pickle('../IEEE_Fraud_Detection/input/train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/input/test.pkl')
    df_train = define_indexes(df_train)
    df_test = define_indexes(df_test)
    df_train = make_featuter(df_train)
    df_test = make_featuter(df_test)
    df_train,df_test = make_featuter2(df_train,df_test)
    df_train,df_test = one_val(df_train,df_test)

    df_train.to_pickle('../IEEE_Fraud_Detection/src/make_data/data/020_train.pkl')
    df_test.to_pickle('../IEEE_Fraud_Detection/src/make_data/data/020_test.pkl')

if __name__ == "__main__":
    main()