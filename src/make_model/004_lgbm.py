# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
# https://www.kaggle.com/andrew60909/lgb-starter-r
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

def objective(X_train,target,trial):
    num_liaves = trial.suggest_int('num_leaves',300,600)
    min_child_weight = trial.suggest_loguniform('min_child_weight',0.1,0.3)
    feature_fraction = trial.suggest_loguniform('feature_fraction',0.2,0.5)
    bagging_fraction = trial.suggest_loguniform('bagging_fraction',0.3,0.6)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf',50,150)
    max_depth = trial.suggest_int('max_depth',5,20)
    learning_rate = trial.suggest_loguniform('learning_rate',0.001,0.01)
    bagging_seed = trial.suggest_int('bagging_seed',5,15)
    reg_alpha = trial.suggest_loguniform('reg_alpha',0.1,0.6)
    reg_lambda = trial.suggest_loguniform('reg_lambda',0.3,0.8)

    params = {
            'num_leaves': num_liaves,
            'min_child_weight': min_child_weight,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_data_in_leaf': min_data_in_leaf,
            'objective': 'binary',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            "boosting_type": "gbdt",
            "bagging_seed": bagging_seed,
            "metric": 'auc',
            "verbosity": -1,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': 47
        }

    train,test,y_tarin,y_test = train_test_split(X_train,target,train_size = 0.7)
    trn_data = lgb.Dataset(train, label=y_tarin)
    val_data = lgb.Dataset(test, label=y_test)

    lgb_model = lgb.train(params,
                            trn_data,
                            10000,
                            valid_sets = [trn_data, val_data],
                            verbose_eval=500,
                            early_stopping_rounds=500)
    pred = lgb_model.predict(test)
    return (1 - roc_auc_score(y_test, pred))

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

    # ハイパラ調整
    study = optuna.create_study()
    study.optimize(functools.partial(objective,X_train,target),n_trials=100)
    print(study.best_params)
    with open('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/param/opt_lgb_param','wb') as f:
        pickle.dump(study.best_params)
    # params = {
    #         'num_leaves': 491,
    #         'min_child_weight': 0.03454472573214212,
    #         'feature_fraction': 0.3797454081646243,
    #         'bagging_fraction': 0.4181193142567742,
    #         'min_data_in_leaf': 106,
    #         'objective': 'binary',
    #         'max_depth': -1,
    #         'learning_rate': 0.006883242363721497,
    #         "boosting_type": "gbdt",
    #         "bagging_seed": 11,
    #         "metric": 'auc',
    #         "verbosity": -1,
    #         'reg_alpha': 0.3899927210061127,
    #         'reg_lambda': 0.6485237330340494,
    #         'random_state': 47
    #     }
    splits = 10
    folds = KFold(n_splits = splits)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))

    log_file = '../IEEE_Fraud_Detection/src/LOG/073_lgb.log'
    logging.basicConfig(filename=log_file)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):
        print("Fold {}".format(fold_))
        logging.critical("Fold {}".format(fold_))

        train_df, y_train_df = X_train.iloc[trn_idx], target.iloc[trn_idx]
        valid_df, y_valid_df = X_train.iloc[val_idx], target.iloc[val_idx]

        trn_data = lgb.Dataset(train_df, label=y_train_df)
        val_data = lgb.Dataset(valid_df, label=y_valid_df)

        lgb_model = lgb.train(study.best_params,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds=500)

        pred = lgb_model.predict(valid_df)
        oof[val_idx] = pred
        auc_score = roc_auc_score(y_valid_df, pred)
        print( "  auc = ", auc_score )
        logging.critical("  auc = " + str(auc_score))

        predictions += lgb_model.predict(X_test) / splits

    filename = '../IEEE_Fraud_Detection/model/073_lgb.sav'
    pickle.dump(lgb_model,open(filename,'wb'))

    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    sub['isFraud'] = predictions
    sub.to_csv('../IEEE_Fraud_Detection/output/073_sub_lgb.csv',index=False)
    print('end')
    cmd = "curl -X POST https://hooks.slack.com/services/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 -d \"{'text': %s }\"" % (auc_score)
    subprocess.run(cmd,shell=True)
    submit_cmd = 'kaggle competitions submit -c ieee-fraud-detection -f ../IEEE_Fraud_Detection/output/073_sub_lgb.csv -m "optunaでハイパラ調整"'
    subprocess.run(submit_cmd,shell=True)


if __name__ == "__main__":
    main()