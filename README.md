# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```c
## Developed by: varsha s
## RegisterNumber: 212222220055

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```
## Output:
## Placement Data:
![image](https://github.com/user-attachments/assets/084df230-7ec2-4245-a6f3-5e2d7a75cc33)

## Salary Data:
![image](https://github.com/user-attachments/assets/965d8401-7f34-48a1-a9e9-12884c100fa9)

## Checking the null() function:

![image](https://github.com/user-attachments/assets/97790fa6-5fb2-41a9-93e5-b78907e23c81)

## Data Duplicate:
![image](https://github.com/user-attachments/assets/858b9bc7-a93a-4ed7-a266-e8c97bd9b458)

## Print Data:
![image](https://github.com/user-attachments/assets/93a14d93-43e7-402e-8912-a2f66a477491)


## Data-Status:
![image](https://github.com/user-attachments/assets/9e35f491-f0fb-4c6a-94c0-4ba4604ced2a)

## Y_prediction array:
![image](https://github.com/user-attachments/assets/b08314f0-62cb-4e39-8838-e804ab40c94e)

## Accuracy value:
![image](https://github.com/user-attachments/assets/cad727b2-e9c9-440e-8dbf-84d8bd0fcbe8)

## Confusion array:
![image](https://github.com/user-attachments/assets/6fa9af21-f821-4164-98cd-d6498362751a)

## Classification Report:

![image](https://github.com/user-attachments/assets/6f873445-278d-4923-a95b-af2eff932456)

## Prediction of LR:
![image](https://github.com/user-attachments/assets/93986047-9013-4e13-8be0-39a09c23e450)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
