# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:15:28 2016

@author: S622970
"""

import math
import pandas as pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
Import and remove useless variables
"""
train = pandas.read_csv("C:/Users/S622970/.ipython/titanic/train.csv", dtype={"Age": np.float64}, )
test = pandas.read_csv("C:/Users/S622970/.ipython/titanic/test.csv", dtype={"Age": np.float64}, )
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)


""" 
transform variables
"""

#Embarked
train["Embarked"] = train["Embarked"].fillna("S")
embark_dummies_train  = pandas.get_dummies(train['Embarked'])
embark_dummies_test  = pandas.get_dummies(test['Embarked'])
train = train.join(embark_dummies_train)
test    = test.join(embark_dummies_test)
train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)


# Fare
train["Fare"].fillna(train["Fare"].mean(), inplace=True)
test["Fare"].fillna(test["Fare"].mean(), inplace=True)
train['Fare'] = train['Fare'].astype(int)
test['Fare']    = test['Fare'].astype(int)

# Age

means_age=train[['Pclass', 'Sex','Age']]
means_age=means_age.groupby(["Pclass", "Sex"]).mean()

i=0
for row in train.iterrows():
    if math.isnan(row[1]['Age']):
        train.set_value(i,'Age',means_age['Age'].ix[row[1]['Pclass']].ix[row[1]['Sex']])
    i=i+1

i=0
for row in test.iterrows():
    if math.isnan(row[1]['Age']):
        test.set_value(i,'Age',means_age['Age'].ix[row[1]['Pclass']].ix[row[1]['Sex']])
    i=i+1


train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)


# Family

train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train = train.drop(['SibSp','Parch'], axis=1)
test    = test.drop(['SibSp','Parch'], axis=1)

# Sex

sex_dummies_train  = pandas.get_dummies(train['Sex'])
sex_dummies_train.columns = ['Female','Male']

sex_dummies_test  = pandas.get_dummies(test['Sex'])
sex_dummies_test.columns = ['Female','Male']

train = train.join(sex_dummies_train)
test    = test.join(sex_dummies_test)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

#PClass

pclass_dummies_train  = pandas.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']


pclass_dummies_test  = pandas.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']


train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_train)
test  = test.join(pclass_dummies_test)

#Cabin


cabins_surv=train[train.Survived==1]
people_by_cabin=(train["Cabin"].value_counts())
people_by_cabin=people_by_cabin.rename("all")
survived_by_cabin=(cabins_surv["Cabin"].value_counts())
survived_by_cabin=survived_by_cabin.rename("survived")

corres=pandas.concat([people_by_cabin, survived_by_cabin], axis=1)
corres["survived"] = corres["survived"].fillna(0)

corres['proba_surv'] =  (corres["survived"] / corres["all"])*100

corres['proba_surv']    = corres['proba_surv'].astype(int)
corres.drop("all",axis=1,inplace=True)
corres.drop("survived",axis=1,inplace=True)


train1=train[(pandas.notnull(train.Cabin))]
train2=train[pandas.isnull(train.Cabin)]

test1=test[(pandas.notnull(test.Cabin))]
test2=test[pandas.isnull(test.Cabin)]

train2.drop("Cabin",axis=1,inplace=True)
test2.drop("Cabin",axis=1,inplace=True)



train1["proba_cabin"]=0

index=0
for row in train1.iterrows():
    train1.set_value(index,"proba_cabin",corres['proba_surv'][row[1].ix['Cabin']])
    index=index+1  
    
test1["proba_cabin"]=0

index=0
for row in test1.iterrows():
    if any(row[1].ix['Cabin']==s for s in corres.index.values):
        test1.set_value(index,"proba_cabin",corres['proba_surv'][row[1].ix['Cabin']])
        index=index+1

train1=train1[(pandas.notnull(train1.Fare))]
test1=test1[(pandas.notnull(test1.Fare))]
    
train1.drop("Cabin",axis=1,inplace=True)
test1.drop("Cabin",axis=1,inplace=True)
            

"""
Run Model
"""
X1_train = train1.drop("Survived",axis=1)
Y1_train = train1["Survived"]
X1_test  = test1.drop("PassengerId",axis=1).copy()

X2_train = train2.drop("Survived",axis=1)
Y2_train = train2["Survived"]
X2_test  = test2.drop("PassengerId",axis=1).copy()

# Random Forests
random_forest2 = RandomForestClassifier(n_estimators=100)

random_forest2.fit(X2_train, Y2_train)

Y2_pred = random_forest2.predict(X2_test)

print("random forests2:")
print(random_forest2.score(X2_train, Y2_train))

random_forest1 = RandomForestClassifier(n_estimators=100)
random_forest1.fit(X1_train, Y1_train)

Y1_pred = random_forest1.predict(X1_test)

print("random forests1:")
print(random_forest1.score(X1_train, Y1_train))

Y1_pred=pandas.DataFrame(Y1_pred)
Y2_pred=pandas.DataFrame(Y2_pred)
Y_pred = [Y1_pred, Y2_pred]
Y_pred=pandas.concat(Y_pred)
Y_pred=list(map(int,Y_pred.ix[:, 0].tolist()))

test1.drop("proba_cabin",axis=1,inplace=True)

test_total=[test1, test2]
test_total=pandas.concat(test_total)


submission = pandas.DataFrame({
        "PassengerId": test_total["PassengerId"].astype(int),
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
