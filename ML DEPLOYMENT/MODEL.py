import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('deceptive-opinion.csv')
df.head()
df.tail()
#Extracting only the requireed features
df1 = df[['deceptive', 'text']]
print(df1)
df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
print(df1)

X = df1['text']
Y = np.asarray(df1['deceptive'],dtype = int)

#importing MultinomialNB
from sklearn.naive_bayes import MultinomialNB, GaussianNB
#splitting the data into training and testing set  with test size is 30%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test
print(X_test)
print(y_test)
nb = MultinomialNB()
#Converting the review (text feature) to numerical features
cv = CountVectorizer()
x = cv.fit_transform(X_train)
y = cv.transform(X_test)
# Fitting the model
nb.fit(x, y_train)
pickle.dump(nb,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
nb.predict(y)
# Training Accuracy
print(nb.score(x, y_train))
# Testing Accuracy
print(nb.score(y, y_test))

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(x, y_train)

#Predict the response for test dataset
y_pred = clf.predict(y)
print(y_pred)
print(clf.score(x, y_train)) #Training accuracy
print(clf.score(y, y_test))