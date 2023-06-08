# Archivo de funciones para aplicar los algortimos de clasificación

import pandas as pd
import numpy as np
import seaborn as sns # Realización de gráficos
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from Analisis_clasificacion import *

# Las siguientes funciónes nos devuelven la matriz de confusión, la curva ROC y también una dataframe con la clasificación de las métricas, 
# tanto para el conjunto de train como para el conjunto de test en cada modelo correspondiente y con los respectivos parámetros.


# Regresión Logística
from sklearn.linear_model import LogisticRegression

def RegresionLogistica(iter, X_train_bal, y_train_bal, X_train, y_train, X_test, y_test, lista):
    logreg = LogisticRegression(max_iter=iter)
    logreg.fit(X_train_bal, y_train_bal)
    logistic_regression_y_pred = logreg.predict(X_test.dropna())
    string = "Regresión Logística"

    lista = metricas_clasificación(lista, logreg, X_train, y_train, y_test, logistic_regression_y_pred,string)

    fig = matriz_confusion_curva_ROC(logreg, logistic_regression_y_pred, X_test, y_test, string)
    return fig, lista

# Arbol de decisión
from sklearn.tree import DecisionTreeClassifier
def ArbolDecision(depth, X_train_bal, y_train_bal, X_train, y_train, X_test, y_test, lista):
    dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=depth, splitter = 'best', random_state=1)
    dt_classifier.fit(X_train_bal, y_train_bal)
    dt_y_pred = dt_classifier.predict(X_test.dropna())
    string = "Árbol de decisión simple"

    lista = metricas_clasificación(lista, dt_classifier, X_train, y_train, y_test, dt_y_pred, string)

    fig = matriz_confusion_curva_ROC(dt_classifier, dt_y_pred, X_test, y_test, string)
    return fig, lista

# Random Forest
from sklearn.ensemble import RandomForestClassifier
def RandomForest(n, X_train_bal, y_train_bal, X_train, y_train, X_test, y_test, lista):
    rf_classifier = RandomForestClassifier(max_depth=5, n_estimators=n, max_features=11, criterion='gini')
    rf_classifier.fit(X_train_bal, y_train_bal)
    rf_y_pred = rf_classifier.predict(X_test.dropna())
    string = "Random Forest Classifier"

    lista = metricas_clasificación(lista, rf_classifier, X_train, y_train, y_test, rf_y_pred, string)

    fig = matriz_confusion_curva_ROC(rf_classifier, rf_y_pred, X_test, y_test, string)
    return fig, lista


# K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
def KVecinosCercanos(n, X_train_bal, y_train_bal, X_train, y_train, X_test, y_test, lista):
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_train_bal, y_train_bal)
    knn_y_pred = knn_classifier.predict(X_test.dropna())
    string = "K-NN Classifier"

    lista = metricas_clasificación(lista, knn_classifier, X_train, y_train, y_test, knn_y_pred, string)

    fig = matriz_confusion_curva_ROC(knn_classifier, knn_y_pred, X_test, y_test, string)
    return fig, lista

# XGBoost Classifier
import xgboost as xgb
def XGBoost(n, X_train_bal, y_train_bal, X_train, y_train, X_test, y_test, lista):
    xgb_classifier = xgb.XGBClassifier(n_estimators=n, objective='binary:logistic')
    xgb_classifier.fit(X_train_bal, y_train_bal)
    xgb_y_pred = xgb_classifier.predict(X_test.dropna())
    string = "XGBoost Classifier"

    lista = metricas_clasificación(lista, xgb_classifier, X_train, y_train, y_test, xgb_y_pred, string)

    fig = matriz_confusion_curva_ROC(xgb_classifier, xgb_y_pred, X_test, y_test, string)
    return fig, lista
