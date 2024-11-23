# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
STEP 1 : Start

STEP 2 : Import and Load Data: Load the student placement dataset using pandas.

STEP 3 : Preprocess Data: Copy the dataset, then drop irrelevant columns like "sl_no" and "salary" to prepare for training.

STEP 4 : Check Data Integrity: Check for missing values and duplicated rows in the cleaned dataset.

STEP 5 : Define Features and Labels: Separate the independent variables (features) and the dependent variable (target) 'status'.

STEP 6 : Split the Data: Split the dataset into training and testing sets using an 80/20 ratio.

STEP 7 : Train the Model: Initialize and train a Logistic Regression model on the training data.

STEP 8 : Evaluate the Model: Predict using the test data, calculate accuracy, generate the classification report, and test with new input.

STEP 9 : End
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARSHA S
RegisterNumber:  212222220055
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset.head()
dataset.info()

dataset = dataset.drop('sl_no', axis=1);
dataset.info()

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes


dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x = dataset.iloc[:,:-1]
x

y=dataset.iloc[:,-1]
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score, confusion_matrix
cf = confusion_matrix(y_test, y_pred)
cf

accuracy=accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Placement Data :
![image](https://github.com/user-attachments/assets/d7a9bca2-ba40-4fc2-9b15-52e88e7aa6b0)
## Salary Data :
![image](https://github.com/user-attachments/assets/f11b4bad-9971-4743-8a7d-805395fd1651)
## Checking the null() function :
![image](https://github.com/user-attachments/assets/83abc147-e2f1-4af2-8188-ad69f412fa54)
## Data Duplicate :
![image](https://github.com/user-attachments/assets/63dda5bb-c0d2-4042-bb0c-fd805e98e3a2)
## Clean Data :
![image](https://github.com/user-attachments/assets/c05455b7-4279-4ac3-b095-6ee20f330d9a)
## Y-Prediction Array :
![image](https://github.com/user-attachments/assets/f7c25ad4-5e81-4d98-bafc-adb95e90bc7a)
## Missing Values Check :
![image](https://github.com/user-attachments/assets/fb9c5782-0b08-4953-9149-5132b25540a5)
## Accuracy value :
![image](https://github.com/user-attachments/assets/4f0e9014-3864-4456-aa48-ab2a1324eddd)
## Confusion array :
![image](https://github.com/user-attachments/assets/d7e0f1c4-a447-413e-86b9-8c3722c22df5)
## Classification Report :
![image](https://github.com/user-attachments/assets/f9dfce0b-608d-4aed-8de9-9fbbe28e96ba)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
