import numpy as np
import pandas as pd






def drop_col(df):
    # V_col https://www.kaggle.com/c/ieee-fraud-detection/discussion/108707#latest-627396
    drop_list = ['V107','V108','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137',
                'V306','V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317',
                'V318','V319','V320','V321']
    df = df.drop(drop_list,axis=1)
    return df

def use_col(df):
    use_list = ['TransactionAmt','ProductCD','card1','card2','card4','card6','P_emaildomain','R_emaildomain',]


def main():
    df_train = pd.read_pickle('../IEEE_Fraud_Detection/input/train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/input/test.pkl')
    df_train = drop_col(df_train)
    df_test = drop_col(df_test)


if __name__ == "__main__":
    main()