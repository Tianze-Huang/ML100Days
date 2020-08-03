# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:47:47 2020

@author: Han
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

df_train=pd.read_csv('house_train.csv.gz')

train_Y=np.log1p(df_train['SalePrice'])
df=df_train.drop(['Id','SalePrice'],axis=1)
df.head()

num_features=[]
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype=='float64' or dtype=='int64':
        num_features.append(feature)

print(f'{len(num_features)} Numeric Features : {num_features}\n')

df=df[num_features]
df=df.fillna(-1)
MMEncoder=MinMaxScaler()
train_num=train_Y.shape[0]
print(df.head())
    
import seaborn as sns 
import matplotlib.pyplot as plt

sns.regplot(x=df['1stFlrSF'][:train_num],y=train_Y)
plt.show()

train_X=MMEncoder.fit_transform(df)
estimator=LinearRegression()
print(cross_val_score(estimator,train_X,train_Y,cv=5).mean())

df['1stFlrSF']=df['GrLivArea'].clip(800,2500)
sns.regplot(x=df['1stFlrSF'],y=train_Y)
plt.show()

train_X=MMEncoder.fit_transform(df)
estimator=LinearRegression()
print(cross_val_score(estimator,train_X,train_Y,cv=5).mean())

print('\n\n\n')

keep_indexs=(df['1stFlrSF']>800)&(df['1stFlrSF']<2500)
df=df[keep_indexs]
train_Y=train_Y[keep_indexs]
sns.regplot(x=df['GrLivArea'],y=train_Y)
plt.show()

train_X=MMEncoder.fit_transform(df)
estimator=LinearRegression()
print(cross_val_score(estimator,train_X,train_Y,cv=5).mean())



