from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, TextField, DecimalField, validators
from wtforms.validators import DataRequired
from wtforms import Form, SubmitField
import pickle
import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

def trimMinAndSecFromDates(dates):
    result = []
    for date in dates:
        result.append(date[:-6])
    return result

def train_classify():
    data = pd.read_csv('train.csv')
    DayOfWeek_dummies = pd.get_dummies(data['DayOfWeek'])
    data = data.join(DayOfWeek_dummies)
    PdDistrict_dummies = pd.get_dummies(data['PdDistrict'])
    data = data.join(PdDistrict_dummies)

    data['Dates'] = trimMinAndSecFromDates(data['Dates'])

    # encode Dates using label encoding
    data['Dates'] = data['Dates'].astype('category')
    data['Dates_int'] = data['Dates'].cat.codes

    # encode Address using label encoding
    data['Address'] = data['Address'].astype('category')
    data['Address_int'] = data['Address'].cat.codes
    data = data.drop(["Resolution", "PdDistrict", "Descript", "DayOfWeek", "Address", "Dates"], axis = 1)
    data = data.drop(data[(data['Y'] > 37.84) | (data['Y'] < 37.7)].index)
    data = data.drop(data[((data['X'] > -122.32) | (data['X'] < -122.52))].index)

    X = data.drop('Category', axis = 1)
    y = data['Category'].copy()

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 8),n_estimators = 40,learning_rate = 0.5, random_state = 1)
    clf.fit(X, y)

    crime_classification_path = 'crime_classification.pkl'
    crime_classification = open(crime_classification_path, 'wb')
    pickle.dump(clf, crime_classification)
    crime_classification.close()

train = train_classify()

def unpickle():
    crime_classification_path = 'crime_classification.pkl'
    model_crime_classification = open(crime_classification_path, 'rb')
    clf_new = pickle.load(model_crime_classification)
    return clf_new

class CrimeClassification(Form):
    submit = SubmitField("Send")

@app.route("/")
def input():
    form = CrimeClassification(request.form)
    return render_template('user_input.html', form=form)


@app.route('/results', methods=['POST'])
def result():
    form = CrimeClassification(request.form)
    if request.method == 'POST' and form.validate():

        longitude = request.form['longitude']
        latitude = request.form['latitude']
        dayWeek = request.form['dayWeek']
        address = request.form['address']
        district = request.form['district']

        d = {'longitude': [longitude], 'latitude': [latitude], 'address': [address], 'date': [pd.to_datetime('1950-12-31')],'district': [district], 'Sunday': [0], 'Monday': [0], 'Tuesday': [0], 'Wednesday': [0], 'Thursday': [0], 'Friday': [0], 'Saturday': [0], 'BAYVIEW': [0], 'CENTRAL': [0],
       'INGLESIDE': [0], 'MISSION': [0], 'NORTHERN': [0], 'PARK': [0], 'RICHMOND': [0], 'SOUTHERN': [0],'TARAVAL': [0], 'TENDERLOIN': [0]}
        d[dayWeek] = [1]
        d[district] = [1]
        df = pd.DataFrame(data=d)
        df["address"] = df["address"].astype('category')
        df["district"] = df["district"].astype('category')
        df["date"] = df["date"].astype('category')
        df["Dates_int"] = df["date"].cat.codes
        df['Address_int'] = df['address'].cat.codes
        df = df.drop(['address', 'district', 'date'], axis = 1)

        review = unpickle()
        result = review.predict(df)

        return render_template('results.html',
                                longitude = longitude, latitude = latitude, dayWeek = dayWeek, address = df['Address_int'], district = 'district', result = result)
