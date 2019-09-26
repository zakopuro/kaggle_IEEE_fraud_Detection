import pandas as pd
import subprocess

# df = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/025_train.pkl')
# print(df['card2_cut'].head())
a = 11
# cmd = "curl -X POST https://hooks.slack.com/services/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 -d \"{'text': %s }\"" % (a)
cmd = 'kaggle competitions submit -c champs-scalar-coupling -f ../IEEE_Fraud_Detection/output/001_sub_xgb.csv -m "test"'
subprocess.run(cmd,shell=True)

# curl -X POST https://hooks.slack.com/services/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 -d '{"text":"test"}'