# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 01:55:28 2020

@author: dips
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
train_df.info()
train_df.isna()

percent_missing=train_df.isnull().sum()*100/len(train_df)
percent_missing1=train_df.isnull().sum()*100/len(test_df)

data=[train_df,test_df]

train_df=train_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
test_df=test_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
train_df.info()

percent_missing2=train_df.isnull().sum()*100/len(train_df)
percent_missing3=train_df.isnull().sum()*100/len(test_df)

mean = train_df['LotFrontage'].mean()
train_df['LotFrontage']=train_df['LotFrontage'].fillna(mean)
mean1 = test_df['LotFrontage'].mean()
test_df['LotFrontage']=test_df['LotFrontage'].fillna(mean1)

data=[train_df,test_df]
train_df['FireplaceQu'].value_counts()
test_df['FireplaceQu'].value_counts()
common='Gd'
for data in data:
    data['FireplaceQu']=data['FireplaceQu'].fillna(common)

train_df['MasVnrType'].value_counts()
test_df['MasVnrType'].value_counts()
print("Skew is:", train_df.SalePrice.skew())
plt.hist(train_df.SalePrice,color='blue')
plt.show()

numeric=train_df.select_dtypes(include=[np.number])
numeric.dtypes

categoricals=train_df.select_dtypes(exclude=[np.number])
categoricals.describe()

train_df['enc_street']=pd.get_dummies(train_df.Street,drop_first=True)
test_df['enc_street']=pd.get_dummies(test_df.Street,drop_first=True)

def encode(x):
    return 1 if x=='Partial' else 0
train_df['enc_condition']=train_df.SaleCondition.apply(encode)
test_df['enc_condition']=test_df.SaleCondition.apply(encode)

data1=train_df.select_dtypes(include=[np.number]).interpolate().dropna()
data2=test_df.select_dtypes(include=[np.number]).interpolate().dropna()

y = np.log(train_df.SalePrice)
X = data1.drop(['SalePrice', 'Id'], axis=1)


X_train = X
Y_train = data1["SalePrice"]
X_test  = data2

from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, Y_train)
X_test=data2.drop(['Id'],axis=1)
predictions = model.predict(X_test)















