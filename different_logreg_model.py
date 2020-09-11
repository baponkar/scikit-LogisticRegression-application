import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
print(df['male'])

kf = KFold(n_splits=5,shuffle=True)

x1 = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
x2 = df[['Pclass','male','Age']].values
x3 = df[['Fare','Age']].values
y = df['Survived'].values

def score_model(x,y,kf):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index,test_index in kf.split(x):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
        model = LogisticRegression(solver='lbfgs')
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test,y_pred))
        precision_scores.append(precision_score(y_test,y_pred))
        recall_scores.append(recall_score(y_test,y_pred))
        f1_scores.append(f1_score(y_test,y_pred))

    print("accuracy:",np.mean(accuracy_scores))
    print("precision:",np.mean(precision_scores))
    print("recall:",np.mean(recall_scores))
    print("f1 score:",np.mean(f1_scores))

print("\nLogistic Regression with all features:")
score_model(x1,y,kf)
print("\nLogistic Regression with Pclass,Sex & Age features:")
score_model(x2,y,kf)
print("\nLogistic Regression with Fare and Age features:")
score_model(x3,y,kf)

