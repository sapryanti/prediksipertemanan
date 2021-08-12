#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import xlsxwriter


# In[2]:


filename="karate.csv"
delimiter=" "
graphname = "socfb-Reed98"

if os.path.isfile(filename) == True:
    # Load data from tsv file: data
    loaddataset = np.loadtxt(filename, delimiter=delimiter, dtype=int)
else:
    loaddataset = 'File does not exist'
loaddataset = loaddataset[:, [0, 1]]
print(loaddataset)

# LoadData = pd.DataFrame(loaddataset, columns=['Node A','Node B'])
# with pd.ExcelWriter('socfb-Reed98.xlsx', engine='openpyxl' , mode = 'a') as writer :
#             LoadData.to_excel(writer, sheet_name='Node A & B')


# In[3]:


print(type(loaddataset))


# In[4]:


G_dataset=nx.Graph()
G_dataset.name = f"Dataset {graphname}"
for u,v in loaddataset:
    if u == v:
        print("u and v are equal : "+str(u)+"&"+str(u))
    else:
        G_dataset.add_edge(u,v)


# In[65]:



# In[5]:


pos = nx.spring_layout(G_dataset)
plt.figure(figsize=(12,8))#3:2
plt.title(f"Graph Dataset {graphname}")
nx.draw(G_dataset,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)


# In[6]:


print(len(set(nx.non_edges(G_dataset))))
print(G_dataset.number_of_nodes())
print(G_dataset.number_of_edges())
print(nx.density(G_dataset))


# In[7]:


print(list(G_dataset.nodes))


# In[8]:


print(list(G_dataset.edges))


# In[9]:





# In[10]:


print(list(nx.non_edges(G_dataset)))


# In[ ]:



akurasi = []
akurasiBayes = []
akurasiDC = []
akurasiRF = []
akurasiSVM = []
inkre = 1
alledges = list(nx.non_edges(G_dataset))+list(G_dataset.edges)
kf = KFold(n_splits=5, random_state=None, shuffle=True) 
for training_index, testing_index in kf.split(alledges):
    print('===============')

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
    #print(pd.DataFrame(y_testing, columns=['Class']))
    ytest = pd.DataFrame(y_testing, columns=['Class_ytest'])

    plt.figure(figsize=(12,8))#3:2
    plt.title(f"Graph Dataset Training-{inkre} {graphname}")
    nx.draw(G_dataset,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
    plt.savefig(f"yanti-Sampling-dataset"+str(i)+".png")

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
    print(isix)
    
    isiy = pd.merge(ytrain, ytest, left_on=['Class_ytrain'],
                    right_on=['Class_ytest'])
    print(isiy)
    
    print(hasilknn,y_testing)
    akurasi.append(accuracy_score(hasilknn,y_testing))

    print('akurasi KNN : ', akurasi)
    report = classification_report(hasilknn,y_testing)
    print('report KNN : ', report)
    
    print('==============')
    
    print(hasilbayes,y_testing)
    akurasiBayes.append(accuracy_score(hasilbayes,y_testing))

    print('akurasi Bayes : ', akurasiBayes)
    report = classification_report(hasilbayes,y_testing)
    print('report Bayes : ', report)
    
    print('==============')
    
    print(hasilDc,y_testing)
    akurasiDC.append(accuracy_score(hasilDc,y_testing))

    print('akurasi DC : ', akurasiDC)
    report = classification_report(hasilDc,y_testing)
    print('report DC : ', report)
    
    print('==============')
    
    print(hasilRF,y_testing)
    akurasiRF.append(accuracy_score(hasilRF,y_testing))

    print('akurasi RF : ', akurasiRF)
    report = classification_report(hasilRF,y_testing)
    print('report RF : ', report)
    
    print('==============')
    
    print(hasilsvm,y_testing)
    akurasiSVM.append(accuracy_score(hasilsvm,y_testing))

    print('akurasi SVM : ', akurasiSVM)
    report = classification_report(hasilsvm,y_testing)
    print('report SVM : ', report)
    
    

#         report = pd.DataFrame(classification_report(hasilknn, y_testing, output_dict=True)).transpose()
# #         report.to_csv('Your Classification Report Name.csv', index= True)


#         with pd.ExcelWriter('dataset.xlsx', engine='openpyxl' , mode = 'a') as writer :
#             isix.to_excel(writer, sheet_name='xtrain-test')
#         with pd.ExcelWriter('dataset.xlsx', engine='openpyxl' , mode = 'a') as writer :
#             isiy.to_excel(writer, sheet_name='ytrain-test')
#         with pd.ExcelWriter(excel, engine='openpyxl' , mode = 'a') as writer :
#             report.to_excel(writer, sheet_name='confusion matrix')

    inkre = inkre+1

#     akurasi = pd.DataFrame(akurasi)

#     with pd.ExcelWriter('hasilruning.xlsx', engine='openpyxl' , mode = 'a') as writer :
#         akurasi.to_excel(writer, sheet_name='akurasi')



# In[ ]:





# In[ ]:




