from flask import Flask, flash, render_template, request, url_for, redirect
import urlib.request
import os
import pandas as pd
import csv
import xlsxwriter
import numpy as numpy



app = Flask(__name__)

class Linkprediction:

    def loaddata():
    filename="karate.csv"
    delimiter=" "
    graphname = "Karate"

    if os.path.isfile(filename) == True:
        # Load data from tsv file: data
        loaddataset = np.loadtxt(filename, delimiter=delimiter, dtype=int)
    else:
        loaddataset = 'File does not exist'
    loaddataset = loaddataset[:, [0, 1]]
    print(loaddataset)


    def split():
        kf = KFold(n_splits=5, random_state=None, shuffle=True) 

    def result():
    wb = load_workbook(filename='E:\Experiment - Skripsi\\Hasil Experiment\\result.xlsx')
    sheet_ranges = wb['cr','cr1','cr2']
    df = pd.DataFrame(sheet_ranges.values)
    da df.dropna()
    da = df.iloc[1:230]
    da.columns = ['iterasi','akurasi','presisi','recall','f1-score']


# def loaddata():
#     filename="karate.csv"
#     delimiter=" "
#     graphname = "Karate"

# if os.path.isfile(filename) == True:
#     # Load data from tsv file: data
#     loaddataset = np.loadtxt(filename, delimiter=delimiter, dtype=int)
# else:
#     loaddataset = 'File does not exist'
# loaddataset = loaddataset[:, [0, 1]]
# print(loaddataset)
if __name__ == "__main__":
    app.run(debug=True)