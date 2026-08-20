"""Canonical script generated from preprocessing_v1.1.ipynb."""

import warnings 
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline

#
filename = '/Users/kaanguney.keklikci/Desktop/Erasmus+/Heidelberg/data/ThoraricSurgery.csv'
df = pd.read_csv(filename)
df.head()

#
df.drop('id',axis=1,inplace=True)

#
df.dtypes

#
sns.countplot(df.DGN,palette='Set3')
sns.despine()

#
dgn_types = ['DGN1','DGN8', 'DGN5', 'DGN6']
df = df[~df.DGN.isin(dgn_types)]

sns.countplot(df.DGN,palette='Set3')
sns.despine()

#
X, y = df.drop('Risk1Yr',axis=1), df.Risk1Yr

#
# ### Training, validation split

#
X_train,X_test,y_train,y_test = train_test_split(X, 
                                                 y,
                                                 test_size=.33,
                                                 random_state=42,
                                                 shuffle=True,
                                                 stratify=y)

#
keepdims = ['DGN','PRE4','PRE5']
X_train, X_test = X_train[keepdims], X_test[keepdims]

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

#
# ### Standardize data

#
scaler = StandardScaler()
scaler.fit(X_train[keepdims[1:]])
scaler

#
X_train = np.append(X_train.DGN.to_numpy().reshape(-1,1), scaler.transform(X_train[keepdims[1:]]), axis=1)
X_test = np.append(X_test.DGN.to_numpy().reshape(-1,1), scaler.transform(X_test[keepdims[1:]]), axis=1)

#
X_train = pd.DataFrame(X_train, columns=keepdims)
X_test = pd.DataFrame(X_test, columns=keepdims)

#
X_train.head()

#
X_test.head()

#
# end of preprocessing

#
