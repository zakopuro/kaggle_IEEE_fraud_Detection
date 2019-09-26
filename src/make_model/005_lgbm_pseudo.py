import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import lightgbm as lgb
import pickle
import logging
import subprocess
import optuna
import functools
import warnings
warnings.filterwarnings('ignore')

def adversarial_val(train,test):
    param = {'num_leaves': 50,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.2,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}

    train['target'] = 0
    test['target'] = 1
    train_test = pd.concat([train,test],axis=0)
    x = train_test.drop('target',axis=1)
    y = train_test['target'].values
    X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42, shuffle=True)

    while True:
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)

        num_round = 100
        lgb_model = lgb.train(
                    param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=50,
                    early_stopping_rounds = 50
                    )

        pred = lgb_model.predict(X_test)
        try:
            auc = roc_auc_score(y_test, pred)
        except ValueError:
            auc = 1

        if auc < 0.80:
            break
        importance = pd.DataFrame(lgb_model.feature_importance(), index=X_train.columns, columns=['importance']).sort_values(by='importance',ascending=False)
        list_drop = list(importance.index[:5])
        X_train = X_train.drop(list_drop,axis=1)
        X_test = X_test.drop(list_drop,axis=1)

    feature = X_train.columns
    print('use feature')
    print(feature)
    train = train.loc[:,feature]
    test = test.loc[:,feature]

    return train,test



def main():
    submit_num = '089'
    overfit_sub = pd.read_csv('../IEEE_Fraud_Detection/src/make_model/overfit/overfit_sub2.csv')

    df_train = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/030_train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/030_test.pkl')

    X_train = df_train.drop(['isFraud','TransactionDT'],axis=1)
    target = df_train['isFraud']
    X_test = df_test.drop('TransactionDT',axis=1)
    del df_train,df_test
    for f in tqdm(X_train.select_dtypes(include='category').columns.tolist() + X_train.select_dtypes(include='object').columns.tolist()):
        lbl = LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

    params = {
            'num_leaves': 491,
            'min_child_weight': 0.03454472573214212,
            'feature_fraction': 0.3797454081646243,
            'bagging_fraction': 0.4181193142567742,
            'min_data_in_leaf': 106,
            'objective': 'binary',
            'max_depth': -1,
            'learning_rate': 0.006883242363721497,
            "boosting_type": "gbdt",
            "bagging_seed": 11,
            "metric": 'auc',
            "verbosity": -1,
            'reg_alpha': 0.3899927210061127,
            'reg_lambda': 0.6485237330340494,
            'random_state': 47
        }
    kfold_splits = 10
    folds = KFold(n_splits=kfold_splits)
    # oof = np.zeros(len(X_train))

    log_file = '../IEEE_Fraud_Detection/src/LOG/'+submit_num+'_lgb.log'
    logging.basicConfig(filename=log_file)

    test_split = 5
    pred_threshold = 0.5
    X_test_index = np.array_split(X_test.index,test_split)
    all_pred = []
    XX_train = X_train.copy()

    for i in range(test_split):
        print(str(i)+'step')
        X_test_split = X_test.loc[X_test_index[i]]
        # ad_val
        X_train_ad,X_test_split_ad = adversarial_val(XX_train,X_test_split)
        predictions = np.zeros(len(X_test_split_ad))
        if i != 0:
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_ad.values, target.values)):
                print("Fold {}".format(fold_))
                logging.critical("Fold {}".format(fold_))
                train_df, y_train_df = X_train_ad.iloc[trn_idx], target.iloc[trn_idx]
                valid_df, y_valid_df = X_train_ad.iloc[val_idx], target.iloc[val_idx]

                trn_data = lgb.Dataset(train_df, label=y_train_df)
                val_data = lgb.Dataset(valid_df, label=y_valid_df)

                lgb_model = lgb.train(params,
                                trn_data,
                                10000,
                                valid_sets = [trn_data, val_data],
                                verbose_eval=500,
                                early_stopping_rounds=500)
                pred = lgb_model.predict(valid_df)
                # oof[val_idx] = pred
                try:
                    auc_score = roc_auc_score(y_valid_df, pred)
                    print( "  auc = ", auc_score )
                    logging.critical("  auc = " + str(auc_score))
                except ValueError:
                    print('auc valueerror')
                    logging.critical(" auc valueerror")
                predictions += lgb_model.predict(X_test_split_ad) / kfold_splits
        else:
            predictions = overfit_sub['isFraud'][:len(X_test_split_ad)].values
        all_pred.extend(predictions)
        if i+1 < test_split:
            df_pred = pd.DataFrame({'isFraud':predictions}, index=X_test_split.index)
            df_pred.loc[df_pred['isFraud'] >pred_threshold,'isFraud'] = 1
            df_pred.loc[df_pred['isFraud'] <=pred_threshold,'isFraud'] = 0
            X_train = pd.concat([X_train,X_test_split])
            target = pd.concat([target,df_pred['isFraud']])
            X_train = X_train[len(X_test_split):]
            target = target[len(X_test_split):]
            target = target.astype(np.int64)

    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    sub['isFraud'] = all_pred
    sub.to_csv('../IEEE_Fraud_Detection/output/'+submit_num+'_sub_lgb.csv',index=False)
    print('end')
    cmd = "curl -X POST https://hooks.slack.com/services/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 -d \"{'text': %s }\"" % (auc_score)
    subprocess.run(cmd,shell=True)
    submit_cmd = 'kaggle competitions submit -c ieee-fraud-detection -f ../IEEE_Fraud_Detection/output/'+submit_num+'_sub_lgb.csv -m "pseudo labeling & ad_val"'
    subprocess.run(submit_cmd,shell=True)

if __name__ == "__main__":
    main()