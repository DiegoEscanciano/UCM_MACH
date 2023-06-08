# Archivo de funciones del análisis exploratorio.

# Importamos los paquetes necesarios para el análisis exploratorio.

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
import numpy as np

# La función Tabla_accidentes_var es una función tal que dada una variable y el dataframe devuelve un dataframe contando las apariciones de # var agrupadas por los valores de var.
# Además, si mortal es igual a 0, este gráfico se hace en función de los accidentes mortales y en caso contrario de los totales.

def Tabla_accidentes_var(var,df,mortal):
    if mortal == 0:
        df_cnt = df.loc[df.severidad == 0].groupby([var])[var].count()
        v_aux = pd.DataFrame(df_cnt)
        v_aux = v_aux.set_axis(['Num_accidentes_mortales'], axis=1, inplace=False)
        v_aux = v_aux.reset_index(level=0)
        v_aux['Porcentaje_accidentes_mortales'] = v_aux['Num_accidentes_mortales']/sum(v_aux['Num_accidentes_mortales'])*100
    else:
        df_cnt = df.groupby([var])[var].count()
        v_aux = pd.DataFrame(df_cnt)
        v_aux = v_aux.set_axis(['Num_accidentes_totales'], axis=1, inplace=False)
        v_aux = v_aux.reset_index(level=0)
        v_aux['Porcentaje_accidentes_totales'] = v_aux['Num_accidentes_totales']/sum(v_aux['Num_accidentes_totales'])*100

    return v_aux.sort_index()

# La función Lineas_accidentes_var es una función tal que dada una variable y el dataframe devuelve un gráfico de líneas contando las apariciones de var agrupadas por los valores de var.
# Además, si mortal es igual a 0 , este gráfico se hace en función de los accidentes mortales y en caso contrario de los totales.

def Lineas_accidentes_var(var,df,mortal):
    df_aux = Tabla_accidentes_var(var,df,mortal)
    plt.figure(figsize = (12,6))
    if mortal == 0:
        sns.lineplot(x = df_aux[var], y = df_aux['Num_accidentes_mortales'], color = 'red')
        plt.title("Número de accidentes mortales según {string}".format(string=var))
        plt.xlabel(var)
        plt.ylabel("Número de accidentes mortales")
    else:
        sns.lineplot(x = df_aux[var], y = df_aux['Num_accidentes_totales'], color = 'dodgerblue')
        plt.title("Número de accidentes totales según {string}".format(string=var))
        plt.xlabel(var)
        plt.ylabel("Número de accidentes totales")

# La función Barras_accidentes_var es una función tal que dada una variable y el dataframe devuelve un gráfico de barras contando el número # de ocurrencias de la variable "var" agrupadas por los valores de var.
# Además, si mortal es igual a 0, este gráfico se hace en función de los accidentes mortales y en caso contrario de los totales.

def Barras_accidentes_var(var,df,mortal):
    df_aux = Tabla_accidentes_var(var,df,mortal)
    plt.figure(figsize = (12,6))
    if mortal == 0:
        sns.barplot(x = df_aux[var], y = df_aux['Num_accidentes_mortales'], color = 'red')
        plt.title("Número de accidentes mortales según {string}".format(string=var))
        plt.xlabel(var)
        plt.ylabel("Número de accidentes mortales")
    else:
        sns.barplot(x = df_aux[var], y = df_aux['Num_accidentes_totales'], color = 'dodgerblue')
        plt.title("Número de accidentes totales según {string}".format(string=var))
        plt.xlabel(var)
        plt.ylabel("Número de accidentes totales")