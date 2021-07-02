from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')

vectorizer = TfidfVectorizer()
wordnet = WordNetLemmatizer()

app = Flask(__name__)
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
clf=pickle.load(open('clf.sav','rb'))



@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/detect',methods=['POST','GET'])

  
def predict():
    
    features =request.form["u_data"]

    words = re.sub('[^a-zA-Z]', ' ',features)
    words = words.lower()
    words = words.split()
    words = [wordnet.lemmatize(word) for word in words if not word in stopwords.words('english')]
    words = ' '.join(words)
    
  
    
    prediction = clf.predict(loaded_vectorizer.transform([words]))
    
       
    if prediction == 1:
        return render_template('index.html',pred='The news is Fake')
    if prediction == 0:
        return render_template('index.html',red='not Fake news')



if __name__ == '__main__':
    app.run(debug=True)
    app.run(host="0.0.0.0", port="33")

