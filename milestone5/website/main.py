from flask import Blueprint,request
from flask import render_template
from flask_login import login_required, current_user
from milestone3 import ZoomFeatureClassifier
from milestone4 import WebexFeatureClassifier
import pandas as pd
import os

model=None
clf=None
old=""
new=""
output=""
main = Blueprint('main', __name__)
@main.route('/')
def index():
    return render_template('index.html',loggedin=current_user.is_authenticated)

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html',count=current_user.count, fname=current_user.fname,lname=current_user.lname, email=current_user.email, loggedin=current_user.is_authenticated)

@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', loaded=False, loggedin=current_user.is_authenticated)


@main.route('/predict')
@login_required
def defpredit():
    return render_template('dashboard.html', loaded=False, loggedin=current_user.is_authenticated)

@main.route('/dashboard', methods=['POST'])
@login_required
def loadModel():
    global model
    global new
    global old
    global clf
    kernel = (request.form.get('kernel'))
    dataset = (request.form.get('dataset'))
    if(dataset=="webex"):
        print("webex")
        model=WebexFeatureClassifier()
    else:
        print("zoom")
        model=ZoomFeatureClassifier()
    X,y=model.load()
    clf,old,new=model.model(kernel,X,y)
    print(model.predict(clf,"new feature"))

    return render_template('dashboard.html',text="",loaded=True, old=", ".join(old), new=", ".join(new),loggedin=current_user.is_authenticated)

@main.route('/predict', methods=['POST'])
@login_required
def predict():
    global output
    text=(request.form.get('text'))
    output=model.predict(clf,text)
    if len(text)>30:
        text=text[0:30]+".."
    return render_template('dashboard.html',output=output,text=text,loaded=True, old=", ".join(old), new=", ".join(new),loggedin=current_user.is_authenticated)
