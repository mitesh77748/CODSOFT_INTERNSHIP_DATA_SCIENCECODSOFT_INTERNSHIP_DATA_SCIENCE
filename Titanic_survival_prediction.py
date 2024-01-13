import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
np.set_printoptions(suppress=True,precision=6)

df=pd.read_csv(r'E:\lab\project\Data_Science\Titanic-Dataset.csv')

df=df.drop(['PassengerId','Name','SibSp','Parch','Ticket',"Cabin",'Embarked'], axis=1)
#print(df) 

target=df['Survived']
inputs=df.drop('Survived',axis=1)

dummies=pd.get_dummies(inputs['Sex'])
inputs=pd.concat([inputs,dummies], axis=1)
inputs=inputs.drop(['Sex'],axis=1)

inputs["Age"].fillna(inputs['Age'].mean(), inplace=True)

X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)
model=GaussianNB()
model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print('Model Accuracy:',accuracy)

pred=model.predict(X_test)
pred_probability=model.predict_proba(X_test)
print(pred[:5])
for i in range (5):
    print(pred_probability[i][0],end=', ')