import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split,TimeSeriesSplit

def main():
    df_train = pd.read_csv('../IEEE_Fraud_Detection/src/make_data/data/005_train.csv',index_col='TransactionID')
    df_test = pd.read_csv('../IEEE_Fraud_Detection/src/make_data/data/005_test.csv',index_col='TransactionID')
    # fti = pd.read_csv('../IEEE_Fraud_Detection/src/make_data/data/fti_list.csv')
    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    print('data')
    # fti_drop_list = list(fti['feat'].head(30))
    X_train = df_train.drop(['isFraud','TransactionDT'],axis=1)
    # X_train = X_train.drop(fti_drop_list,axis=1)
    y_train = df_train['isFraud'].copy()
    X_test = df_test.copy()
    X_test = X_test.drop(['TransactionDT'],axis=1)
    # X_test = X_test.drop(fti_drop_list,axis=1)


    xgb_paramas = {
                    'n_estimators':500,
                    'max_depth':9,
                    'learning_rate':0.05,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    'missing':-999,
                    'gamma':0.2,
                    'alpha':4,
                    'tree_method':'hist',
                    'eval_metric': 'auc'
    }
        # n_estimators=500,
        # max_depth=9,
        # learning_rate=0.05,
        # subsample=0.9,
        # colsample_bytree=0.9,
        # gamma = 0.2,
        # alpha = 4,
        # missing = -1,
        # tree_method='gpu_hist'


    # xgb_model = xgb.XGBClassifier(**xgb_paramas)
    # xgb_params = {
    #     "n_estimators": 2000,
    #     "seed": 4,
    #     # "silent": True,
    #     "max_depth": 9,
    #     "learning_rate": 0.03,
    #     "subsample": 0.9,
    #     "colsample_bytree": 0.9,
    #     "tree_method": "hist",
    #     # "objective": "binary:logistic",
    #     'missing':-999,
    #     'gamma':0.2,
    #     'alpha':4,
    #     "eval_metric": "auc"
    # }




    # 時系列データの交差検証
    # train_start_index = 0
    # train_end_index = int(len(df_train)/2)
    # skip = int(len(df_train)/6)
    # horizon = int(len(df_train)/6)
    # SPLITS = 3
    # for _ in range(SPLITS):
    #     test_start_index = train_end_index
    #     test_end_index = test_start_index + horizon

    #     X_train = df_train[train_start_index:train_end_index]
    #     X_test = df_train[test_start_index:test_end_index]

    #     train_start_index += skip
    #     train_end_index += skip

    # xgb_model.fit(X_train,y_train)
    n_fold = 5
    # folds = TimeSeriesSplit(n_splits=n_fold)
    folds = KFold(n_splits=5)
    # EPOCHS = 3
    # kf = KFold(n_splits = EPOCHS, shuffle = True)
    y_preds = np.zeros(sub.shape[0])
    y_oof = np.zeros(X_train.shape[0])
    for tr_idx, val_idx in folds.split(X_train, y_train):
        xgb_model = xgb.XGBClassifier(**xgb_paramas)

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        xgb_model.fit(X_tr,y_tr)
        y_pred_train = xgb_model.predict_proba(X_vl)[:,1]
        y_oof[val_idx] = y_pred_train
        print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))

        y_preds+= xgb_model.predict_proba(X_test)[:,1] / n_fold

    print('fit')
    filename = '../IEEE_Fraud_Detection/model/012_xgb.sav'
    pickle.dump(xgb_model,open(filename,'wb'))
    # sub['isFraud'] = xgb_model.predict_proba(X_test)[]
    # sub['isFraud'] = xgb_model.predict_proba(X_test)[:,1]
    sub['isFraud'] = y_preds
    sub.to_csv('../IEEE_Fraud_Detection/output/012_sub_xgb.csv',index=False)
    print('end')


if __name__ == "__main__":
    main()