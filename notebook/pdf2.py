# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: mlpy36
#     language: python
#     name: mlpy36
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing
import pandas_profiling as pdf

sub = pd.read_csv('../IEEE_Fraud_Detection/input/sample_submission.csv')

df_train = pd.read_csv('../IEEE_Fraud_Detection/src/make_data/data/001_train.csv')

pdf.ProfileReport(df_train)


