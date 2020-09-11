"""
This is a machine learning practice program.
The tutorials taken from SOLOLEARN android app.
Author : Mr. Bapon Kar
Build Date : 11/09/2020
Last Updated : 11/09/2020
Version : v-1.0.0
Description : In this program uses 'titanic.csv'download link
              (https://sololearn.com/uploads/files/titanic.csv)
              as data source which has
              [Survived,Pclass,Sex,Age,Siblings/Spouces,Parents/Children
              Fare] columns.Survived has two data 1[True] and 0[False].
              It has total 887 row and 7 columns features.

              In this program I am using logisticRegression model
              to train and predict the data.Logisticregression model
              doesnot return just a prediction ,but it returns a probability
              level from 0 to 1.It has following parameters
              [penalty,dual,tol,C,fit_intercept,intercept_scalling,
              class_weight,random_state,solver,max_iter,multi_class,verbose,
              warm-start,n_jobs,l1_ratio].
              
              My python version is python-3.6.9 and
              Sci-kit learning version-0.20.2

References :  [1] A. C. Muller and S. Guido - Introduction to Machine
                  Learning with Python - 2017
              [2] Hands on Machine Learning with Scikit Learn and Tensorflow
              [3] https://www.sololearn.com
              [4] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_recall_fscore_support,precision_score
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
x = df[[ 'Age','Fare']].values 
y = df['Survived'].values
print("Input data shape:",x.shape)
print("Output data shape :",y.shape)

#Splitting the initial data into 75: 25 ratio into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,random_state=0)

#Making LogisticRegression model and train with training data
lr = LogisticRegression(solver='lbfgs')
lr.fit(x_train,y_train)

#Analysis
print("Test Score :",lr.score(x_test,y_test))
print("Training score:",lr.score(x_train,y_train))
#This model using straight line lr.coef_[0]*x + lr.coef_[1]*y + intercept = 0 
# for future prediction
print("Coefficient :",lr.coef_)
print("Intercept :",lr.intercept_)



# Precision = TP/(TP+FP);fraction of TP in total positive prediction
# precision is a measure of how precise the model is with its positive predictions
# Recall =  TP/(TP+FN) ;fraction of TP in total actual positive
# Recall : Recall is a measure of how many of the positive cases the model can recall.
# F1 Score = 2.(precision.recall)/(precision + recall)
# The F! score is the harmonic mean of the precision and recall values
#TP=True Positive,TN=True Negative,FP=False positive,FN=False Negative



y_pred = lr.predict(x)
sensitivity_score = recall_score
def specificity_score(y_true, y_pred):
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
    return r[0]
print("Sensitivity : ",sensitivity_score(y,y_pred))
print("Speciificity : ",specificity_score(y,y_pred))
print("Predicted Probability :",lr.predict_proba(x_test)[:,1])

#Giving threshold value =0.75
y_pred = lr.predict_proba(x_test)[:,1] > 0.75
#print("Precision score:",precision_score(y_test,lr.predict(x_test)))
#print("recall score : ",recall_score(y_test,lr.predict(x_test)))
print("Precision score:",precision_score(y_test,y_pred))
print("recall score : ",recall_score(y_test,y_pred))





#Confusion matrix
# Elements
# a11 = Actual negative which predict as negative(TN)
# a12 = Actually negative which predicted as positive(FP)
# a21 = Actually positive which predicted as negative(FN)
# a22 = Actually positive which predicted as positive (TP)

from sklearn.metrics import confusion_matrix
output = confusion_matrix(y,lr.predict(x))
print("Confusion matrix :\n",output)

# Now showing scatter graph with predicted model straight line
# which divided two class say Survived or not survived
x_points = x[:,0]
y_points = [ (-lr.coef_[0][0]*i - lr.intercept_)/lr.coef_[0][1] for i in x[:,0] ]
plt.scatter(x[:,0],x[:,1],label="data")
plt.plot(x_points,y_points,label="predicted line")
plt.legend(loc='best')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
#in the figure you can clearly show two sets of data one in upperside of straight 
#line and other in below of the straight line
#actually we can differentiate this two colors by different color with pyplot(long)
# or with mglearn(short code).But I did not show that in here




#Output:
#   Output data shape : (887,)
#   Test Score : 0.6666666666666666
#   Training score: 0.6526315789473685
#   Coefficient : [[-0.01514699  0.01426569]]
#   Intercept : [-0.484297]
#   Sensitivity :  0.21929824561403508
#   Speciificity :  0.9302752293577982
#   Predicted Probability : [0.47392086 0.41565892 0.30447107 0.32869253 0.46839903 0.32674178
# 0.30494323 0.55853921 0.47196251 0.29797591 0.33513016 0.65087323
#0.32594796 0.29225527 0.31133858 0.34195833 0.33477566 0.32738445
#0.34537486 0.21386476 0.43975092 0.38742187 0.31239463 0.38820762
#0.32037347 0.29810007 0.36919779 0.69559857 0.41053715 0.37940749
# 0.76602938 0.30707267 0.34260368 0.2267947  0.31136135 0.30725146
# ....
# 0.25608672 0.35597262 0.51985077 0.33224803 0.2972702  0.26316387
# 0.73410047 0.44188967 0.31253942 0.34616722 0.5557742  0.47088837]
# Precision score: 0.6666666666666666
# recall score :  0.023255813953488372

#   Confusion matrix :
#  [[507  38]
#   [267  75]]






