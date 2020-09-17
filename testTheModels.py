import numpy as np
import time
import pandas as pd
from numpy import genfromtxt
import csv
from numpy import loadtxt
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
import logging

logging.basicConfig(filename='log_testTheModels.log', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

Y = np.load("labels_2.npy")
X = np.load("matrices_2.npy", allow_pickle=True)
withoutDuplicates = list(dict.fromkeys(Y))

for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=42)

    nsamples, nx, ny = X_train.shape
    X_train_2D = X_train.reshape((nsamples, nx*ny))

    scaler = StandardScaler()
    scaler.fit(X_train_2D)
    X_train_2D = scaler.transform(X_train_2D)

    seed = 7
    models = []
    models.append(('LR', LogisticRegression(max_iter = 1000)))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))

    logging.warning(" ........................ testTheModels.py ........................ ")
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        start = time.time()
        logging.warning("============= %s model =============", name)
        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train_2D, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        end = time.time()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        logging.warning("Elapsed time: %f", end - start)
        logging.warning(msg)

    #boxplot algorithm comparison
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # #plt.show()
    # saveTime = time.time()
    # plt.savefig(fname = "Results_"+str(saveTime)+".png")
