"""Canonical script generated from preprocessing_v1.0.ipynb."""

import warnings 
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt 

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
# ### Define the encoder

#
ordinal_encoder = OrdinalEncoder()

ordinal_encoder

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
# ### Encode data

#
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)

#
X_train.dtypes

#
# encode training 
X_train = pd.DataFrame(ordinal_encoder.fit_transform(X_train), columns=X_train.columns)
X_train.head()

#
# encode testing 
X_test = pd.DataFrame(ordinal_encoder.fit_transform(X_test), columns=X_test.columns)
X_test.head()

#
X_test.dtypes

#
# ### Standardize data

#
scaler = StandardScaler()
scaler.fit(X_train)

#
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#
X_train.head()

#
X_test.head()

#
# end of preprocessing

#
