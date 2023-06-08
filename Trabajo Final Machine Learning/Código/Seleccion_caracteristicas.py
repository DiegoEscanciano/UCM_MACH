# Archivo de funciones para la selección de características

import pandas as pd
import numpy as np

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import f_classif, mutual_info_classif
import matplotlib.pyplot as plt

# Función que selecciona una muestra y muestra por pantalla la distribución de la muestra creando un dataframe auxiliar
def df_Muestra(df, n, var_obj):
    df_muestra = df.sample(n)

    df_sample_var = pd.DataFrame(df_muestra.groupby([var_obj])[var_obj].count().rename('Count'))
    df_sample_var['Porcentaje'] = df_sample_var['Count']/df_sample_var['Count'].sum()

    df_sample_var = df_sample_var.sort_values(by = "Count", ascending=False).reset_index()
    print(df_sample_var)
    return df_muestra

# Función que balancea una muestra y muestra por pantalla la distribución de las etiquetas antes y después del balanceo
def Balanceo(df_muestra, var_obj):
    oversample =  RandomOverSampler(sampling_strategy=0.6)
    X = df_muestra.drop([var_obj], axis=1)
    y = df_muestra[var_obj]

    X_bal, y_bal = oversample.fit_resample(X, y)
    df_sample_bal = X_bal
    df_sample_bal['severidad'] = y_bal

    print ("Distribución de las estiquetas antes del 'resampling:' {}".format(Counter(y)))
    print ("Distribución de las estiquetas después del 'resampling:' {}".format(Counter(y_bal)))
    return df_sample_bal

# Función que muestra el gráfico de la selección de caracteristicas del dataframe proporcionado en función de la variable objetivo.
# Emplea dos métricas: F-Test Score y Mutual Information Score.
def Seleccion_carac(df, var_obj):
    # convertimos el DataFrame al formato necesario para scikit-learn
    data = df.values
    X = df.drop([var_obj], axis=1)
    y = df[var_obj] 
    feature_names = X.columns

    f_test, _ = f_classif(X, y)
    f_test /= np.max(f_test)

    mi = mutual_info_classif(X, y)
    mi /= np.max(mi)

    plt.figure(figsize=(20, 5))
    plt.subplot(1,2,1)
    plt.bar(range(X.shape[1]),f_test,  align="center")
    plt.xticks(range(X.shape[1]),feature_names, rotation = 90)
    plt.xlabel('features')
    plt.ylabel('Ranking')
    plt.title('$F-Test$ score')

    plt.subplot(1,2,2)
    plt.bar(range(X.shape[1]),mi, align="center")
    plt.xticks(range(X.shape[1]),feature_names, rotation = 90)
    plt.xlabel('features')
    plt.ylabel('Ranking')
    plt.title('Mutual information score')

    plt.show()

