# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:55:58 2020

@author: dips
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
train_data= pd.read_csv("train (1).csv")
test_data=pd.read_csv("test.csv")
train_data.info()
train_data=train_data.drop(['PassengerId'],axis=1)
train_data=train_data.drop(['Cabin'],axis=1)
train_data=train_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Cabin'],axis=1)

#filling missing values
mean = train_data['Age'].mean()
train_data['Age']=train_data['Age'].fillna(mean)
mean1 = test_data['Age'].mean()
test_data['Age']=test_data['Age'].fillna(mean1)
train_data['Embarked'].describe()

#filling missing values of embarked column
common = 'S'
data=[train_data,test_data]
for data in data:
    data['Embarked']=data['Embarked'].fillna(common)
    
train_data.info() 
data = [train_data, test_data]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_data['not_alone'].value_counts()


#converting fare values from float to integers
data=[train_data,test_data]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

#converting gender into numeric values
data=[train_data,test_data]    
genders = {"male": 0, "female": 1}
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
 
#dropping ticket column 
data=[train_data,test_data]    
train_data['Ticket'].describe()    
train_data = train_data.drop(['Ticket'], axis=1)
test_data = test_data.drop(['Ticket'], axis=1)

#converting embarked into numeric values:
data=[train_data,test_data]
ports = {"S": 0, "C": 1, "Q": 2}
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

#forming age groups:
data=[train_data,test_data]    
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
 
#categorizing fare into three values: 
data=[train_data,test_data]    
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#creating new features:
data=[train_data,test_data]    
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

data=[train_data,test_data]
mean2= test_data['Age_Class'].mean()
test_data['Age_Class']=test_data['Age_Class'].fillna(mean2)

data=[train_data,test_data]
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


#declaring train and test data:
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]    
X_test  = test_data.drop("PassengerId", axis=1).copy()


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


#RANDOM FOREST
#fitting the random forest regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
max_depths = np.linspace(1,10,10,endpoint=True)
for max_depth in max_depths:
    regressor = RandomForestRegressor(n_estimators=512, random_state = 0,max_depth=max_depth)
    regressor.fit(X_train,Y_train)
#predicting new result 
y_pred = regressor.predict(X_test)
regressor.score(X_train,Y_train)
acc_regressor=round(regressor.score(X_train,Y_train)*100,2)


#DECISION TREE:
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state = 0)
regressor1.fit(X_train,Y_train)
# Predicting a new result
y_pred1 = regressor1.predict(X_test)
acc_regressor1=round(regressor1.score(X_train,Y_train)*100,2)












