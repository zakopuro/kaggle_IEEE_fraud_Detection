# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
# https://www.kaggle.com/andrew60909/lgb-starter-r
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime
# import lightgbm as lgb
import xgboost as xgb
import pickle
import logging
import subprocess

def main():
    df_train = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/030_train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/030_test.pkl')
    target = df_train['isFraud'].copy()
    X_train = df_train.drop('isFraud',axis=1)
    X_train.drop(['TransactionDT'],axis=1,inplace=True)
    X_test = df_test.drop(['TransactionDT'],axis=1)
    del df_train
    del df_test

    for f in tqdm(X_train.select_dtypes(include='category').columns.tolist() + X_train.select_dtypes(include='object').columns.tolist()):
        lbl = LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))
    # drop_list = ['V41','id_29','V269','V302','V252','id_12','V31','V195','V334','id_35','V138']
    # X_train.drop(drop_list, axis=1, inplace=True)
    # X_test.drop(drop_list, axis=1, inplace=True)
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    params = {
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        'n_estimators':1000,
        'max_depth':9,
        'learning_rate':0.048,
        'subsample':0.85,
        'colsample_bytree':0.85,
        'missing':-999,
        # tree_method='gpu_hist',  # THE MAGICAL PARAMETER
        'reg_alpha':0.15,
        'reg_lamdba':0.85
        }
    splits = 5
    folds = KFold(n_splits = splits)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))

    log_file = '../IEEE_Fraud_Detection/src/LOG/071_xgb.log'
    logging.basicConfig(filename=log_file)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):
        print("Fold {}".format(fold_))
        logging.critical("Fold {}".format(fold_))

        train_df, y_train_df = X_train.iloc[trn_idx], target.iloc[trn_idx]
        valid_df, y_valid_df = X_train.iloc[val_idx], target.iloc[val_idx]

        trn_data = xgb.DMatrix(train_df, label=y_train_df)
        val_data = xgb.DMatrix(valid_df, label=y_valid_df)

        xgb_model = xgb.train(params=params,
                        dtrain=trn_data,
                        num_boost_round=10000,
                        evals = [(trn_data,'Train'), (val_data,'Val')],
                        verbose_eval=500,
                        early_stopping_rounds=500)

        pred = xgb_model.predict(val_data)
        oof[val_idx] = pred
        auc_score = roc_auc_score(y_valid_df, pred)
        print( "  auc = ", auc_score )
        logging.critical("  auc = " + str(auc_score))

        predictions += xgb_model.predict(xgb.DMatrix(X_test)) / splits

        filename = '../IEEE_Fraud_Detection/model/071_xgb' + str(fold_)+'.sav'
        pickle.dump(xgb_model,open(filename,'wb'))

        del xgb_model
        del pred
        del train_df
        del y_train_df
        del valid_df
        del y_valid_df
        del trn_data
        del val_data


    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    sub['isFraud'] = predictions
    sub.to_csv('../IEEE_Fraud_Detection/output/071_sub_xgb.csv',index=False)
    print('end')
    cmd = "curl -X POST https://hooks.slack.com/services/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 -d \"{'text': %s }\"" % (auc_score)
    subprocess.run(cmd,shell=True)
    submit_cmd = 'kaggle competitions submit -c ieee-fraud-detection -f ../IEEE_Fraud_Detection/output/071_sub_xgb.csv -m "xgboostで試す"'
    subprocess.run(submit_cmd,shell=True)


if __name__ == "__main__":
    main()