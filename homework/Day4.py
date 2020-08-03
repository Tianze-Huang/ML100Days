import os 
import numpy as np
import pandas as pd


f_app=os.path.join('application_train.csv')
print('Path of read in data:%s'%(f_app))
app_train=pd.read_csv(f_app)

row,column=app_train.shape
print('rows:{}\ncolumns:{}'.format(row,column))

j=0
for i in app_train:
    print(i)
    j+=1

print('欄位:%d'%(j))

#節取五到十行的資料
w=app_train.iloc[5:11]
print(w)