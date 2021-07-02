#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

data =pd.read_csv("train.csv")

data.head()

data.shape

data.isnull().sum()

data = data.fillna('')

data.isnull().sum()

data['label'].value_counts()

data['content'] = data['author']+' '+data['title']

data.head()

X = data['content']
y = data['label']

wordnet = WordNetLemmatizer()


def lemmatize(content):
  words = re.sub('[^a-zA-Z]', ' ',content)
  words = words.lower()
  words = words.split()
  words = [wordnet.lemmatize(word) for word in words if not word in stopwords.words('english')]
  words = ' '.join(words)
  return words

data['content'] = data['content'].apply(lemmatize)

data['content']

X = data['content'].values
Y = data['label'].values

print(X.shape , y.shape)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)



"""**y_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train,y_pred)

print('train accuracy score : ', train_accuracy)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test,y_pred)

print('test accuracy score : ', test_accuracy)



confusion_matrix(Y_test,y_pred)

plot_confusion_matrix(model, X_test, Y_test)  
plt.show()


cross_val_score(model, X_test, Y_test, cv=10, scoring = 'accuracy').mean()


print(classification_report(Y_test,y_pred))



print(Y_train[0])

test = X_train[0]

prediction = model.predict(test)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

# data1 =pd.read_csv("/content/drive/MyDrive/datasets/fake-news/train.csv")
# data1 = data.fillna('')
# data1['content'] = data1['author']+' '+data1['title']

# data1['content'][3]

# data1['label'][3]



def classify_message(text):
    text = vectorizer.transform(text)
    predicted = model.predict(text)
    probability = model.predict_proba(text).max()*100

    if predicted==0:
        print("not Fake news")
        print('the probability percentage is:',round(probability))
    else:
        print("The news is Fake")
        print('the probability percentage is:',round(probability))

test = ['Aaron Klein Obamaâ€™s Organizing for Action Partners with Soros-Linked â€˜Indivisibleâ€™ to Disrupt Trumpâ€™s Agenda']
classify_message(test)**"""


import pickle
import os

if not os.path.exists('models'):
    os.makedirs('models')
    
MODEL_PATH = "models/clf.sav"
pickle.dump(model, open(MODEL_PATH, 'wb'))

vec_file = "models/vectorizer.pickle"
pickle.dump(vectorizer, open(vec_file, 'wb'))

