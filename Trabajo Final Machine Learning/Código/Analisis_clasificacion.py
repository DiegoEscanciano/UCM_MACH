# Archivo de funciones auxiliares para el análisis de clasificación.

# Importamos en primer lugar todas las funciones que vamos a utilizar en nuestro trabajo.

import pandas as pd
import numpy as np
import seaborn as sns # Realización de gráficos
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# La función train_test_div_bal es una función que recibe como parámetros un dataset y el nombre de la variable objetivo.
# Esta función divide el dataset proporcionado en conjuntos de entrenamiento y test y devuelve los dos 
# casos de los valores X e y, respectivamente.
# Además, en segundo lugar, se balancean los valores de X_train e y_train con la estrategia "sam_stra".

# Se fija de manera general test_size al 20% y en consecuencia el train _size al 80%
def train_test_div_bal(df,var_obj,sam_stra):
    train, test = train_test_split(df, test_size=0.2, stratify = df[var_obj])
    X_train = train.dropna().drop(var_obj, axis = 1)
    y_train = train.dropna()[var_obj]
    X_test = test.dropna().drop(var_obj, axis = 1)
    y_test = test.dropna()[var_obj]

    oversample = RandomOverSampler(sampling_strategy=sam_stra)
    X_train_bal, y_train_bal = oversample.fit_resample(X_train, y_train)
    for column in X_train_bal.columns:
        X_train_bal[column] = pd.to_numeric(X_train_bal[column], errors = 'ignore', downcast= 'integer')

    return X_train_bal, y_train_bal, X_train, y_train, X_test, y_test

# La función matriz_confusion_curva_ROC, como bien indica su nombre, nos devuelve la matriz de confusión y la curva ROC según el modelo que # se emplee para llevar a cabo la clasificación. Para ello, recibe como parámetros un modelo, model_y_pred e y_test para realizar la 
# comparación , string para ajustar el título de nuestra gráfica de la curva ROC , y también X_test para realizar la predicción. 

def matriz_confusion_curva_ROC(model, model_y_pred, X_test, y_test, string):
    # Calcular la curva ROC en el primer subplot (ax1).

    model_y_pred_prob = model.predict_proba(X_test)[::, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, model_y_pred_prob)
    auc = metrics.roc_auc_score(y_test, model_y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Graficar la curva ROC en el primer subplot (ax1).

    ax1.plot([0, 1], [0, 1], color="#232323")
    ax1.plot(fpr, tpr, label="data 1, auc="+str(auc), color="#ffb83c")
    ax1.set_title('Curva ROC: {}'.format(string))
    ax1.set_xlabel('Tasa Falsos Positivos (FPR)')
    ax1.set_ylabel('Tasa Verdaderos Positivos (TPR)')
    ax1.legend(loc=4)


    # Calcular la matriz de confusión en el segundo subplot (ax2).

    logreg_cf_matrix = confusion_matrix(y_test, model_y_pred)
    
    group_names = ['Verdadero Negativo','Falso Positivo','Falso Negativo','Verdadero Positivo']
    group_counts = ["{0:0.0f}".format(value) for value in logreg_cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in logreg_cf_matrix.flatten()/np.sum(logreg_cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    
    # Graficar la matriz de confusión en el segundo subplot (ax2).

    labels = np.asarray(labels).reshape(2,2)
    ax2 = sns.heatmap(logreg_cf_matrix, annot=labels, fmt='', cmap="OrRd")
    txt = 'Matriz de Confusión: {s}'
    ax2.set_title(txt.format(s=string))
    ax2.set_xlabel('Valores Predecidos')
    ax2.set_ylabel('Valores Reales')
    ax2.xaxis.set_ticklabels(['0','1'])
    ax2.yaxis.set_ticklabels(['0','1'])

    # Agrupamos los dos plot en la misma linea.

    plt.tight_layout()
    plt.show()

    return fig

# Esta función, llamada lista_no_duplicados recibe como parámetro una lista y es una función que elimina las entradas duplicadas de una lista. Es un bucle poco eficiente computacionalmente pero que nos servirá para lo que necesitamos hacer.
# Las listas que utilizaremos son listas de listas de no más de 50 elementos, así pues no debería presentar muchos problemas de optimización.

def lista_no_duplicados(lista):
    lista_no_duplicados=[]
    for x in lista:
        if x not in lista_no_duplicados:
            lista_no_duplicados.append(x)
    return lista_no_duplicados

# Dada una lista (que empezará vacía) esta función devuelve la lista con los datos de las métricas de ese algoritmo para test y train.
# Posteriormente esta lista se convertirá en un dataframe para poder comparar la efectividad de todos los algoritmos.

def metricas_clasificación(list, model, X_train, y_train, y_test, model_y_pred, string):
    
    txt1 = '{s} (Test)'
    txt2 = '{s} (Train)'
    
    log_metrics = [txt1.format(s=string), 
                   metrics.accuracy_score(y_test, model_y_pred), 
                   metrics.precision_score(y_test, model_y_pred), 
                   metrics.recall_score(y_test, model_y_pred), 
                   metrics.f1_score(y_test, model_y_pred), 
                   metrics.roc_auc_score(y_test, model_y_pred)]
    list.append(log_metrics)

    model_y_train_pred = model.predict(X_train.dropna())

    log_train_metrics = [txt2.format(s=string), 
                   metrics.accuracy_score(y_train, model_y_train_pred), 
                   metrics.precision_score(y_train, model_y_train_pred), 
                   metrics.recall_score(y_train, model_y_train_pred), 
                   metrics.f1_score(y_train, model_y_train_pred), 
                   metrics.roc_auc_score(y_train, model_y_train_pred)]
    list.append(log_train_metrics)
    
    return lista_no_duplicados(list)

# La función metrica_tabla recibe por parámetro una lista de la función métrica algoritmo y la transforma en su correspondiente 
# dataframe , cuyas columnas son las indicadas más abajo.

def metrica_tabla(lista):
    metricas_algoritmos_columnas =['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC']
    return pd.DataFrame(lista,columns=metricas_algoritmos_columnas)