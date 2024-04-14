#pre-processing the dataset of credit card fraud detection.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

df = pd.read_csv("C:\\Users\\VISHALBHARDWAJ\\Documents\\_Project_Exhibition_Compressed\\_Project_Exhibition\\dataset\\Dataset\\creditcard.csv")

print(df.head)

#there is no null value so there's no need to add code to remove columns with null values.
#df.dropna()

#df.info()

X = df.values
y_data = df['Class'].values
X = np.delete(X,30,axis=1)

print(X.shape)
print(y_data.shape)
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=0)

classifier = LogisticRegression()
print(classifier.fit(X_train, y_train))
print("\n")

y_pred = classifier.predict(X_test)

print("Confusion Matrix\n")
print(pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(y_pred, name = 'Predicted')))

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average = 'weighted')
    recall = recall_score(y_test, y_predicted, average = 'weighted')
    f1 = f1_score(y_test, y_predicted, average = 'weighted')
    return accuracy, precision, recall, f1

print("\n")

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
