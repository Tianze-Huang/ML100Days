import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f_app=os.path.join('application_train.csv')
print('Path of read in data: {}'.format(f_app))
app_train=pd.read_csv(f_app)
print(app_train.head())

five_num=[0,25,50,75,100]
quantile_5s=[np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],q=i) for i in five_num]
print(quantile_5s)

q_all=app_train['AMT_ANNUITY'][0:101]

pd.DataFrame({'q':list(range(101)),
              'value':q_all})

print(f"Before replace NAs, numbers of row that AMT_ANNUITY is NAs: {sum(app_train['AMT_ANNUITY'].isnull())}")

q_50=quantile_5s[2]
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']=q_50

print(f"After replace NAs, number of row that AMT_ANNUITY is NAs:{sum(app_train['AMT_ANNUITY'].isnull())}")


print("==Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

def normalize_value(x):
    x=((x-min(x))/(max(x)-min(x))-0.5)*2
    return x

app_train['AMT_ANNUITY_NORMALIZED']=normalize_value(app_train['AMT_ANNUITY'])

print("==Normalized data range ==")
print(app_train['AMT_ANNUITY_NORMALIZED'].describe())

print(f"Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs:{sum(app_train['AMT_GOODS_PRICE'].isnull())}")


#眾數
from collections import defaultdict

mode_dict=defaultdict(lambda:0)

for value in app_train[~app_train['AMT_GOODS_PRICE'].isnull()]['AMT_GOODS_PRICE']:
    mode_dict[value]+=1
    
mode_get=sorted(mode_dict.items(),key=lambda kv:kv[1],reverse=True)

value_most=mode_get[0][0]
print(f"The value_most is: {value_most}")

#mode_goods_price=list(app_train['AMT_GOODS_PRICE'].value_counts().index)
#app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(),'AMT_GOODS_PRICES']=mode_goods_price
mode_goods_price=list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(),'AMT_GOODS_PRICE']=mode_goods_price[0]

print(f"After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs:{sum(app_train['AMT_GOODS_PRICE'].isnull())}")
