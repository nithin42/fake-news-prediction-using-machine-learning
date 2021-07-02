from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
clf=pickle.load(open('clf.sav','rb'))



app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/detect',methods=['POST','GET'])

  
def predict():
    
    features =request.form["u_data"]
    data = [features]
    vect = loaded_vectorizer.transform(data).toarray()
    
    prediction = clf.predict(vect)
    
       
    if prediction == 1:
        return render_template('index.html',pred='The news is Fake')
    if prediction == 0:
        return render_template('index.html',red='not Fake news')



if __name__ == '__main__':
    app.run(debug=True)
    app.run(host="0.0.0.0", port="33")

