import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('loan_data.csv')
type(dataset)
dataset.head()
dataset.shape
dataset.describe()
dataset.isnull().sum()
ataset = dataset.dropna()
dataset.isnull().sum()
dataset = dataset.dropna()
dataset.isnull().sum()
dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)
dataset.head()
dataset['Dependents'].value_counts()
dataset.replace({"Dependents":{'3+':4}}, inplace=True)
dataset['Dependents'].value_counts()

sns.countplot(x='Education',hue='Loan_Status',data=dataset)
dataset.replace({"Married":{'No':0,'Yes':1}, "Gender":{'Male':1,'Female':0},"Self_Employed":{'Yes':1,'No':0},"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2},"Education":{'Graduate':1,'Not Graduate':0}}, inplace=True)
dataset.head()
x=dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y=dataset['Loan_Status']
dataset.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2)
model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)
x_test_prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(x_test_prediction,y_test)
accuracy_test
input_data=(1,1,4,1,0,9504,0.0,275.0,360.0,1.0,0)
import numpy as np
input_data_as_numpy=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
if (prediction[0]==0):
    print("you are denied loan")
else :
    print("You are accepted ")
import pickle 
filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))
