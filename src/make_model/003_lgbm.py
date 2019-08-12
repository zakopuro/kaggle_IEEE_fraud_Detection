# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime
import lightgbm as lgb
import pickle


def main():
    df_train = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/009_train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/009_test.pkl')
    target = df_train['isFraud'].copy()
    X_train = df_train.drop('isFraud',axis=1)
    X_train.drop('TransactionDT',axis=1,inplace=True)
    X_test = df_test.drop('TransactionDT',axis=1)

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
    splits = 5
    folds = KFold(n_splits = splits)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):
        print("Fold {}".format(fold_))
        train_df, y_train_df = X_train.iloc[trn_idx], target.iloc[trn_idx]
        valid_df, y_valid_df = X_train.iloc[val_idx], target.iloc[val_idx]

        trn_data = lgb.Dataset(train_df, label=y_train_df)
        val_data = lgb.Dataset(valid_df, label=y_valid_df)

        lgb_model = lgb.train(params,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds=500)

        pred = lgb_model.predict(valid_df)
        oof[val_idx] = pred
        print( "  auc = ", roc_auc_score(y_valid_df, pred) )
        predictions += lgb_model.predict(X_test) / splits

    filename = '../IEEE_Fraud_Detection/model/022_lgb.sav'
    pickle.dump(lgb_model,open(filename,'wb'))

    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    sub['isFraud'] = predictions
    sub.to_csv('../IEEE_Fraud_Detection/output/021_sub_lgb.csv',index=False)
    print('end')


if __name__ == "__main__":
    main()