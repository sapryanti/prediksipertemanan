from flask import Flask, render_template, request,redirect, url_for, flash
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import datetime
from openpyxl import load_workbook
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.metrics import precision_score
from matplotlib.colors import ListedColormap
from sklearn import tree, neighbors, datasets
import xlsxwriter, csv
from werkzeug.utils import secure_filename
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# from wtfrom.validators import InputRequired

UPLOAD_FOLDER = 'static/csv/'
ALLOWED_EXTENSIONS = 'csv'

app = Flask (__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '&^&ffsmwf092;'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('halamanawal.html')

@app.route('/predict')
def predict():
    return render_template('predik.html')


@app.route('/hasil', methods=['GET','POST'])
def hasil():
    delimiter=" "
    graphname = ""
    if request.method == 'POST':
        file = request.files['files']
        if file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("JSO BOSW", file_path)
    loaddataset = np.loadtxt(file_path, delimiter=delimiter, dtype=int, encoding='cp1252')
    loaddataset = loaddataset[:, [0, 1]]
    data = loaddataset

    G_dataset=nx.Graph()
    G_dataset.name = f"Dataset {graphname}"
    for u,v in loaddataset:
        if u == v:
            print("u and v are equal : "+str(u)+"&"+str(u))
        else:
            G_dataset.add_edge(u,v)

    pos = nx.spring_layout(G_dataset)
    plt.figure(figsize=(12,8))#3:2
    plt.title(f"Graph Dataset {graphname}")
    nx.draw(G_dataset,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
    dir_1 = os.path.dirname(__file__)
    dir_2 = os.path.join(dir_1, 'static/img/graph_dataset/')
    gbr_2 = plt.savefig(dir_2 + "Dataset"+".png")

    d1 = (len(set(nx.non_edges(G_dataset))))
    d2 = (G_dataset.number_of_nodes())
    d3 = (G_dataset.number_of_edges())
    d4 = (nx.density(G_dataset))

    d5 = (list(G_dataset.nodes))

    d6 = (list(G_dataset.edges))


    d7 = (list(nx.non_edges(G_dataset)))

    akurasi = [] 
    cfknn=[]
    reknn = []
    preknn = []
    akurasiBayes = []
    akurasiDC = []
    akurasiRF = []
    akurasiSVM = []
    f1knn = []
    inkre = 1
    alledges = list(nx.non_edges(G_dataset))+list(G_dataset.edges)
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    if request.method == 'POST':
        kf = KFold(n_splits=int(request.form['split']), random_state=None, shuffle=True)
    for training_index, testing_index in kf.split(alledges):
        G_train=G_dataset.copy()
        G_train.name = f"Training-{inkre} {graphname}"
        eigenvector = nx.eigenvector_centrality(G_train)

        for i in testing_index:
            k = alledges[i]
            if k in list(G_dataset.edges):
                G_train.remove_edge(k[0], k[1])

        #hitung feature untuk X_training
        x_training = []
        for i in training_index:
            k = alledges[i]
            pa_degree = G_train.degree[k[0]]*G_train.degree[k[1]]
            pa_cluster = nx.clustering(G_train, k[0])*nx.clustering(G_train, k[1])
            pa_closeness = nx.closeness_centrality(G_train, k[0])*nx.closeness_centrality(G_train, k[1])
            pa_eigenvector = eigenvector[k[0]]*eigenvector[k[1]]
            x_training.append([pa_degree, pa_cluster, pa_closeness, pa_eigenvector])
        #print(pd.DataFrame(x_training, columns=['Degree', 'Clustering', 'Closeness', 'Eigen_Vector']))
        xtrain = pd.DataFrame(x_training, columns=['Degree', 'Clustering', 'Closeness', 'Eigen_Vector'])

        #hitung feature untuk X_testing
        x_testing = []
        for i in testing_index:
            k = alledges[i]
            pa_degree = G_train.degree[k[0]]*G_train.degree[k[1]]
            pa_cluster = nx.clustering(G_train, k[0])*nx.clustering(G_train, k[1])
            pa_closeness = nx.closeness_centrality(G_train, k[0])*nx.closeness_centrality(G_train, k[1])
            pa_eigenvector = eigenvector[k[0]]*eigenvector[k[1]]
            x_testing.append([pa_degree, pa_cluster, pa_closeness, pa_eigenvector])
        #print(pd.DataFrame(x_testing, columns=['Degree', 'Clustering', 'Closeness', 'Eigen_Vector']))
        xtest = pd.DataFrame(x_testing, columns=['Degree', 'Clustering', 'Closeness', 'Eigen_Vector'])

        #hitung class untuk y_training
        y_training = []
        for i in training_index:
            k = alledges[i]
            if k in list(G_dataset.edges):
                y_training.append(1)
            else:
                y_training.append(0)
        #print(pd.DataFrame(y_training, columns=['Class']))
        ytrain = pd.DataFrame(y_training, columns=['Class_ytrain'])

        #hitung class untuk y_testing
        y_testing = []
        for i in testing_index:
            k = alledges[i]
            if k in list(G_dataset.edges):
                y_testing.append(1)
            else:
                y_testing.append(0)
            if request.method == 'POST':
                if request.form['pilih'] == 'knn':
                    knn=KNeighborsClassifier(n_neighbors=5)
                    knn.fit(x_training,y_training)
                    hasilknn = knn.predict(x_testing)
                    # print(hasilknn,y_testing)
                    datanya = "KNN"
                elif request.form['pilih'] == 'naive':
                    ModelBayes = GaussianNB()
                    ModelBayes.fit(x_training, y_training)
                    hasilbayes = ModelBayes.predict(x_testing)
                    # print(hasilbayes,y_testing)
                    datanya = "Naive Bayes"
                elif request.form['pilih'] == 'RF':
                    ModelRF = RandomForestClassifier()
                    datnya = ModelRF.fit(x_training, y_training)
                    hasilRF = ModelRF.predict(x_testing)
                    datanya = "RF"
                    _ = tree.plot_tree(datnya, filled=True, fontsize=8, rounded = True)
                    dirr1 = os.path.dirname(__file__)
                    dirr2 = os.path.join(dirr1, 'static/tree/')
                    plt.savefig(dirr2 + "rftree"+".png")
                elif request.form['pilih'] == 'SVM': 
                    modelsvm = SVC()
                    modelsvm.fit(x_training, y_training)
                    hasilsvm = modelsvm.predict(x_testing)
                    datanya = "SVM"
                else:
                    ModelDc = DecisionTreeClassifier()
                    dtnya = ModelDc.fit(x_training, y_training)
                    hasilDc = ModelDc.predict(x_testing)
                    datanya = "DC"
                    clf = tree.DecisionTreeClassifier(random_state=0)
                    clf = clf.fit(xtrain, ytrain)
                    tree.plot_tree(clf)  
                    _ = tree.plot_tree(dtnya, filled=True, fontsize=8, rounded = True)
                    dirr1 = os.path.dirname(__file__)
                    dirr2 = os.path.join(dirr1, 'static/tree/')
                    plt.savefig(dirr2 + "dtree"+".png")
                    
        if request.method == 'POST':
            n_neighbors = int(request.form['split'])
            # # import some data to play with
            # # iris = datasets.load_iris()

            # # we only take the first two features. We could avoid this ugly
            # # slicing by using a two-dim dataset
            # X = x_training
            # y = x_testing

            # siti = ['tidak teman', 'teman']


            # h = .02  # step size in the mesh

            # # Create color maps
            # cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
            # cmap_bold = ['darkorange', 'c', 'darkblue']

            # # we create an instance of Neighbours Classifier and fit the data.
            # clf = neighbors.KNeighborsClassifier(n_neighbors)
            # clf.fit(X, y)

            # # Plot the decision boundary. For that, we will assign a color to each
            # # point in the mesh [x_min, x_max]x[y_min, y_max].
            # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            #                                      np.arange(y_min, y_max, h))
            # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # # Put the result into a color plot
            # Z = Z.reshape(xx.shape)
            # plt.figure(figsize=(8, 6))
            # plt.contourf(xx, yy, Z, cmap=cmap_light)

            # # Plot also the training points
            # sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=siti[y],palette=cmap_bold, alpha=1.0, edgecolor="black")
            # plt.xlim(xx.min(), xx.max())
            # plt.ylim(yy.min(), yy.max())
            # plt.title("3-Class classification (k = %i)" % (n_neighbors))
            # plt.xlabel(siti[0])
            # plt.ylabel(siti[1])
            if n_neighbors==5 :
                dk = plt.savefig('static/img/plot5.png')
            elif n_neighbors==10 :
                dk = plt.savefig('static/img/plot10.png')
                
            
        
        #print(pd.DataFrame(y_testing, columns=['Class']))
        ytest = pd.DataFrame(y_testing, columns=['Class_ytest'])

        pos = nx.spring_layout(G_dataset)
        plt.figure(figsize=(12,8))#3:2
        plt.title(f"Graph Dataset Training-{inkre} {graphname}")
        nx.draw(G_dataset,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
        dir1 = os.path.dirname(__file__)
        dir2 = os.path.join(dir1, 'static/img/graph/')
        gbr = plt.savefig(dir2 + "Graph Dataset Training"+str(i)+".png")

        knn=KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_training,y_training)
        hasilknn = knn.predict(x_testing)
        
        ModelBayes = GaussianNB()
        ModelBayes.fit(x_training, y_training)
        hasilbayes = ModelBayes.predict(x_testing)
        
        ModelDc = DecisionTreeClassifier()
        ModelDc.fit(x_training, y_training)
        hasilDc = ModelDc.predict(x_testing)
        
        ModelRF = RandomForestClassifier()
        ModelRF.fit(x_training, y_training)
        hasilRF = ModelRF.predict(x_testing)
        
        modelsvm = SVC()
        modelsvm.fit(x_training, y_training)
        hasilsvm = modelsvm.predict(x_testing)

        isix = pd.merge(xtrain, xtest, left_on=['Degree'],
                    right_on=['Degree'])
        # print(isix)
        
        isiy = pd.merge(ytrain, ytest, left_on=['Class_ytrain'],
                        right_on=['Class_ytest'])
        # print(isiy)
        # 3
        # print(hasilknn,y_testing)
        akurasi.append(accuracy_score(hasilknn,y_testing))

        # print('akurasi KNN : ', akurasi)
        report = classification_report(hasilknn,y_testing)
        # report = pd.DataFrame(classification_report(hasilknn, y_testing, output_dict=True)).transpose()
        cfknn.append(report)
        # print('report KNN : ', report)

        recall = recall_score(hasilknn,y_testing)
        reknn.append(recall)
        # print('recall : ', recall)
        
        precision = precision_score(hasilknn, y_testing)
        preknn.append(precision)
        # print('precision :', precision)

        f1 = f1_score(hasilknn, y_testing)
        f1knn.append(f1)
        # print('F1-Score :', f1)
        
        
        # print('==============')
        
        # print(hasilbayes,y_testing)
        akurasiBayes.append(accuracy_score(hasilbayes,y_testing))

        # print('akurasi Bayes : ', akurasiBayes)
        report2 = classification_report(hasilbayes,y_testing)
        # print('report Bayes : ', report2)
        
        # print('==============')
        
        # print(hasilDc,y_testing)
        akurasiDC.append(accuracy_score(hasilDc,y_testing))

        # print('akurasi DC : ', akurasiDC)
        report3 = classification_report(hasilDc,y_testing)
        # print('report DC : ', report3)
        
        # print('==============')
        
        # print(hasilRF,y_testing)
        akurasiRF.append(accuracy_score(hasilRF,y_testing))

        # print('akurasi RF : ', akurasiRF)
        report4 = classification_report(hasilRF,y_testing)
        # print('report RF : ', report4)
        
        # print('==============')
        
        # print(hasilsvm,y_testing)

        # print('akurasi SVM : ', akurasiSVM)
        report5 = classification_report(hasilsvm,y_testing)
        # print('report SVM : ', report5)
        inkre = inkre+1
    images = os.listdir(os.path.join(app.static_folder, "img/graph"))
    images2 = os.listdir(os.path.join(app.static_folder, "img/graph_dataset"))
    return render_template('prediksi.html',data=data,gbr=gbr,aq=akurasi, kf=kf, datanya=datanya, ab=akurasiBayes,
                           adc=akurasiDC,arf=akurasiRF, asv=akurasiSVM, re=cfknn, re2=report2,preknn=preknn,f1knn=f1knn,
                           re3=report3, re4=report4, re5=report5,dk=dk,images=images, reknn=reknn,gbr_2=gbr_2,images2=images2)
                

if __name__ == "__main__":
    app.run(debug=True, threaded=True)


