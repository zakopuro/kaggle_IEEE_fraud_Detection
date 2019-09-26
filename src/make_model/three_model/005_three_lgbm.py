# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
# https://www.kaggle.com/andrew60909/lgb-starter-r
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime
import lightgbm as lgb
import pickle
import logging
import subprocess

def main():
    card4_list = ['american express',np.nan,'discover','mastercard','visa']
    sub = pd.DataFrame()
    for card4 in card4_list:
        df_train = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/three_data/031_train' +str(card4)+ '.pkl')
        df_test = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/three_data/031_test' +str(card4)+ '.pkl')
        target = df_train['isFraud'].copy()
        X_train = df_train.drop('isFraud',axis=1)
        X_train.drop(['TransactionDT'],axis=1,inplace=True)
        X_test = df_test.drop(['TransactionDT'],axis=1)

        del df_train
        del df_test

        try:
            np.isnan(card4)
            all_nan_col = [ 'id_07','id_08','id_21','id_22','id_23','id_24','id_25','id_26','id_27','D15_to_mean_card2','D15_to_std_card2']
            X_train.drop(all_nan_col,axis=1,inplace=True)
            X_test.drop(all_nan_col,axis=1,inplace=True)
            print(card4)
        except TypeError:
            if card4 == 'american express':
                all_nan_col = ['dist1','D11','M1','M2','M3','M5','M6','M7','M8','M9','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','dist1_fq_enc']
                X_train.drop(all_nan_col,axis=1,inplace=True)
                X_test.drop(all_nan_col,axis=1,inplace=True)
            print(card4)

        for f in tqdm(X_train.select_dtypes(include='category').columns.tolist() + X_train.select_dtypes(include='object').columns.tolist()):
            lbl = LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values))
        # drop_list = ['V41','id_29','V269','V302','V252','id_12','V31','V195','V334','id_35','V138']
        # X_train.drop(drop_list, axis=1, inplace=True)
        # X_test.drop(drop_list, axis=1, inplace=True)

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
        if card4 == 'discover':
            splits = 3
        elif card4 == 'mastercard':
            splits = 10
        elif card4 == 'visa':
            splits = 10
        elif card4 == 'american express':
            splits = 5
        else:
            splits = 3
        folds = KFold(n_splits = splits)
        oof = np.zeros(len(X_train))
        predictions = np.zeros(len(X_test))

        log_file = '../IEEE_Fraud_Detection/src/LOG/062_lgb' +str(card4)+ '.log'
        logging.basicConfig(filename=log_file)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):
            print("Fold {}".format(fold_))
            logging.critical("Fold {}".format(fold_))

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
            auc_score = roc_auc_score(y_valid_df, pred)
            print( "  auc = ", auc_score )
            logging.critical("  auc = " + str(auc_score))

            predictions += lgb_model.predict(X_test) / splits

        filename = '../IEEE_Fraud_Detection/model/062_lgb' +str(card4) + '.sav'
        pickle.dump(lgb_model,open(filename,'wb'))
        temp_sub = pd.DataFrame(X_test.index)
        temp_sub['isFraud'] = predictions
        sub = pd.concat([sub,temp_sub])


    # sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    # sub['isFraud'] = predictions
    sub = sub.sort_values('TransactionID',ascending=True)
    sub.to_csv('../IEEE_Fraud_Detection/output/062_sub_lgb.csv',index=False)
    print('end')
    cmd = "curl -X POST https://hooks.slack.com/services/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 -d \"{'text': %s }\"" % (auc_score)
    subprocess.run(cmd,shell=True)
    submit_cmd = 'kaggle competitions submit -c ieee-fraud-detection -f ../IEEE_Fraud_Detection/output/062_sub_lgb.csv -m "card4のモデルをそれぞれ作成し結合"'
    subprocess.run(submit_cmd,shell=True)


if __name__ == "__main__":
    main()