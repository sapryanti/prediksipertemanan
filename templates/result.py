from flask import Flask, flash, render_template, request, url_for, redirect
import urlib.request
import os
import pandas as pd
import csv
import xlsxwriter
from werkzeug.utils import secure_filname
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, RadioField
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def result():

wb = load_workbook(filename='E:\Experiment - Skripsi\\Hasil Experiment\\result.xlsx')
sheet_ranges = wb['cr','cr1','cr2']
df = pd.DataFrame(sheet_ranges.values)
da df.dropna()
da = df.iloc[1:230]
da.columns = ['iterasi','akurasi','presisi','recall','f1-score']


if __name__ == "__main__":
    app.run()