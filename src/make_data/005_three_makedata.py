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
    # START_DATE = '2017-12-01'
    # startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    # df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    
    # # df['year'] = df['TransactionDT'].dt.year
    # df['month'] = df['TransactionDT'].dt.month
    # df['dow'] = df['TransactionDT'].dt.dayofweek
    # df['hour'] = df['TransactionDT'].dt.hour
    # df['day'] = df['TransactionDT'].dt.day
   
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

    df['Missing_count'] = df.isna().sum(axis=1)
    df['TransactionAmt_Log'] = np.log(df['TransactionAmt'])
    df['card1/card2'] = np.log(df['card1']/df['card2'])
    df['card1/addr1'] = np.log(df['card1']/df['addr1'])
    df['card2/addr1'] = np.log(df['card2']/df['addr1'])
    df['card1*card2'] = np.log(df['card1'] * df['card2'])
    df['card1*addr1'] = np.log(df['card1'] * df['addr1'])
    df['card1*addr2'] = df['card1'] * df['addr2']
    df['card2*addr1'] = np.log(df['card2'] * df['addr1'])
    df['card2*addr2'] = df['card2'] * df['addr2']

    i_col = ['card1','card2']
    for col in i_col:
        df['TransactionAmt_to_mean_'+col] = df['TransactionAmt'] / df.groupby([col])['TransactionAmt'].transform('mean')
        df['TransactionAmt_to_std_'+col] = df['TransactionAmt'] / df.groupby([col])['TransactionAmt'].transform('std')
        df['id_02_to_mean_'+col] = df['id_02'] / df.groupby([col])['id_02'].transform('mean')
        df['id_02_to_std_'+col] = df['id_02'] / df.groupby([col])['id_02'].transform('std')
        df['D15_to_mean_'+col] = df['D15'] / df.groupby([col])['D15'].transform('mean')
        df['D15_to_std_'+col] = df['D15'] / df.groupby([col])['D15'].transform('std')
    df['D15_to_mean_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('mean')
    df['D15_to_std_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('std')


    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353
    df['uid'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)
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



    ############################################################################
    # ひたすら追加しまくる
    df['Trans_min_mean'] = df['TransactionAmt'] - df['TransactionAmt'].mean()
    df['card1_min_mean'] = df['card1'] - df['card1'].mean()
    df['card2_min_mean'] = df['card2'] - df['card2'].mean()
    df['addr1_min_mean'] = df['addr1'] - df['addr1'].mean()
    df['Trans_min_std'] = df['Trans_min_mean']/df['TransactionAmt'].std()
    df['card1_min_mean'] = df['card1_min_mean']/df['card1'].std()
    df['card2_min_mean'] = df['card2_min_mean']/df['card2'].std()
    df['addr1_min_mean'] = df['addr1_min_mean']/df['addr1'].std()
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
    # card1_cut = [0,2500,5000,7500,10000,12500,15000,17500,20000]
    # df['card1_cut'] = pd.cut(df['card1'],bins=card1_cut)
    # card2_cut = [99,200,300,400,500,600]
    # df['card2_cut'] = pd.cut(df['card2'],bins=card2_cut)
    df['card1-card2'] = df['card1'] - df['card2']
    df['card1+card2'] = df['card1'] + df['card2']
    df['D2*D4'] = df['D2'] * df['D4']
    df['D2+D4'] = df['D2'] - df['D4']
    # df['email_check'] = np.where(df['P_emaildomain']==df['R_emaildomain'],1,0)
    df['email_check_nan_all'] = np.where((df['P_emaildomain'].isna())&(df['R_emaildomain'].isna()),1,0)
    df['email_check_nan_any'] = np.where((df['P_emaildomain'].isna())|(df['R_emaildomain'].isna()),1,0)
    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')
    df['email_match_not_nan'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&
                                     (df['P_emaildomain']!='email_not_provided'),1,0)
    df['P_email_prefix'] = df['P_emaildomain'].apply(lambda x: x.split('.')[0])
    df['R_email_prefix'] = df['R_emaildomain'].apply(lambda x: x.split('.')[0])
    # M_Feature
    M_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']
    df['M_type'] = ''
    for col in M_cols:
        df['M_type'] = '_'+df[col].astype(str)
        df['M_sum'] = df[M_cols].sum(axis=1).astype(np.int8)
        df['M_na'] = df[M_cols].isna().sum(axis=1).astype(np.int8)
    # C_Feature
    C_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
    df['C_sum'] = 0
    df['C_null'] = 0
    for col in C_cols:
        df['C_sum'] += np.where(df[col]==1,1,0)
        df['C_null'] += np.where(df[col]==0,1,0)
        valid_values = df[col].value_counts()
        valid_values = valid_values[valid_values>1000]
        valid_values = list(valid_values.index)
        df[col+'_valid'] = np.where(df[col].isin(valid_values),1,0)
    # Device info
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_c'] = df['DeviceInfo']
    device_match_dict = {
    'sm':'sm-',
    'sm':'samsung',
    'huawei':'huawei',
    'moto':'moto',
    'rv':'rv:',
    'trident':'trident',
    'lg':'lg-',
    'htc':'htc',
    'blade':'blade',
    'windows':'windows',
    'lenovo':'lenovo',
    'linux':'linux',
    'f3':'f3',
    'f5':'f5'
    }
    for dev_type_s, dev_type_o in device_match_dict.items():
        df['DeviceInfo_c'] = df['DeviceInfo_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)
    df['DeviceInfo_c'] = df['DeviceInfo_c'].apply(lambda x: 'other_d_type' if x not in device_match_dict else x)
    # Device info2
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_c'] = df['id_30']
    device_match_dict = {
    'ios':'ios',
    'windows':'windows',
    'mac':'mac',
    'android':'android'
    }
    for dev_type_s, dev_type_o in device_match_dict.items():
        df['id_30_c'] = df['id_30_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)
    df['id_30_v'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
    df['id_30_v'] = np.where(df['id_30_v']!='', df['id_30_v'], 0).astype(int)
    ########################## Anomaly Search in geo information
    df['bank_type'] = df['card3'].astype(str)+'_'+df['card5'].astype(str)
    df['address_match'] = df['bank_type'].astype(str)+'_'+df['addr2'].astype(str)

    return df

def make_featuter2(train,test):
    # https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
    count_col = ['card1','card2','card3','card5','card6','addr1','addr2','card1/card2','card1/addr1','card2/addr1',
                'card1*card2','card1*addr2','card2*addr2','card2*addr1','id_34','id_36']
    for col in count_col:
        train[col+'_count_full'] = train[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
        test[col+'_count_full'] = test[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))

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

    i_cols = ['card1','card2','card3','card5','card1/card2','card1/addr1','card2/addr1',
              'card1*card2','card1*addr1','card1*addr2','card2*addr1','card2*addr2',
              'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
              'D1','D2','D3','D4','D5','D6','D7','D8','D9',
              'addr1','addr2',
              'dist1','dist2',
              'P_emaildomain', 'R_emaildomain',
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



    ############################################################################
    # ひたすら追加しまくる
    for col in ['address_match','bank_type']:
        temp_df = pd.concat([train[[col]], test[[col]]])
        temp_df[col] = np.where(temp_df[col].str.contains('nan'), np.nan, temp_df[col])
        temp_df = temp_df.dropna()
        fq_encode = temp_df[col].value_counts().to_dict()
        train[col] = train[col].map(fq_encode)
        test[col]  = test[col].map(fq_encode)
    train['address_match'] = train['address_match']/train['bank_type']
    test['address_match']  = test['address_match']/test['bank_type']
    drop_list = ['bank_type']
    train.drop(drop_list,axis=1,inplace=True)
    test.drop(drop_list,axis=1,inplace=True)
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again?scriptVersionId=18889353
    # Browser feature
    train['id_31'] = train['id_31'].fillna('unknown_br').str.lower()
    test['id_31']  = test['id_31'].fillna('unknown_br').str.lower()

    train['id_31'] = train['id_31'].apply(lambda x: x.replace('webview','webvw'))
    test['id_31']  = test['id_31'].apply(lambda x: x.replace('webview','webvw'))

    train['id_31'] = train['id_31'].apply(lambda x: x.replace('for',' '))
    test['id_31']  = test['id_31'].apply(lambda x: x.replace('for',' '))

    browser_list = set(list(train['id_31'].unique()) + list(test['id_31'].unique()))
    browser_list2 = []
    for item in browser_list:
        browser_list2 += item.split(' ')
    browser_list2 = list(set(browser_list2))

    browser_list3 = []
    for item in browser_list2:
        browser_list3 += item.split('/')
    browser_list3 = list(set(browser_list3))

    for item in browser_list3:
        train['id_31_e_'+item] = np.where(train['id_31'].str.contains(item),1,0).astype(np.int8)
        test['id_31_e_'+item] = np.where(test['id_31'].str.contains(item),1,0).astype(np.int8)
        if train['id_31_e_'+item].sum()<100:
            del train['id_31_e_'+item], test['id_31_e_'+item]

    train['id_31_v'] = train['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
    test['id_31_v'] = test['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))

    train['id_31_v'] = np.where(train['id_31_v']!='', train['id_31_v'], 0).astype(int)
    test['id_31_v'] = np.where(test['id_31_v']!='', test['id_31_v'], 0).astype(int)
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
    print(cols_to_drop)

    train.drop(cols_to_drop, axis=1, inplace=True)
    test.drop(cols_to_drop, axis=1, inplace=True)

    return train,test


def make_date(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    # df['year'] = df['TransactionDT'].dt.year
    df['month'] = df['TransactionDT'].dt.month
    df['dow'] = df['TransactionDT'].dt.dayofweek
    df['hour'] = df['TransactionDT'].dt.hour
    df['day'] = df['TransactionDT'].dt.day

    return df



def main():
    train = pd.read_pickle('../IEEE_Fraud_Detection/input/train.pkl')
    test = pd.read_pickle('../IEEE_Fraud_Detection/input/test.pkl')
    card4_list = train['card4'].unique()
    train = make_date(train)
    test = make_date(test)
    for i,card4 in enumerate(card4_list):
        if i == 4:
            df_train = train[train['card4'].isnull()]
            df_test = test[test['card4'].isnull()]
        else:
            df_train = train[train['card4'] == card4]
            df_test = test[test['card4'] == card4]
        print(card4)
        df_train.drop('card4', axis=1, inplace=True)
        df_test.drop('card4', axis=1, inplace=True)

        print('読込')
        df_train = define_indexes(df_train)
        df_test = define_indexes(df_test)
        print('index')
        df_train = make_featuter(df_train)
        df_test = make_featuter(df_test)
        print('feature1')
        df_train,df_test = make_featuter2(df_train,df_test)
        # df_train,df_test = one_val(df_train,df_test)
        print('feature2')
        df_train.to_pickle('../IEEE_Fraud_Detection/src/make_data/data/three_data/031_train' +str(card4)+ '.pkl')
        df_test.to_pickle('../IEEE_Fraud_Detection/src/make_data/data/three_data/031_test'  +str(card4)+ '.pkl')

if __name__ == "__main__":
    main()