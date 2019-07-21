import pandas as pd
import xgboost as xgb
import pickle



def main():
    df_test = pd.read_csv('../IEEE_Fraud_Detection/src/make_data/data/001_test.csv')
    sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')
    X_test = df_test.copy()
    filename = '../IEEE_Fraud_Detection/model/001_xgb.sav'
    loaded_model = pickle.load(open(filename,'rb'))
    sub['isFraud'] = loaded_model.predict_proba(X_test)[:,1]
    sub.to_csv('../IEEE_Fraud_Detection/output/001_sub_xgb.csv',index=False)
    print('end')


if __name__ == "__main__":
    main()