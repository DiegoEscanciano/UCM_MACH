# Archivo de funciones para el análisis preliminar y el preprocessing.

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.impute import SimpleImputer # Imputación de valores missing sencilla

# Definimos la función TratamientoDataFrame, que recibe como parámetros un dataframe, unas columnas y columnas_eliminar. Esta función
# renombra las columnas con el valor que se le introduce como parámetro. También se eliminan las columnas introducidas en columnas_eliminar.

def TratamientoDataFrame(df,columnas,columnas_eliminar):
    df.columns = columnas
    df_res = df.drop(columnas_eliminar, axis=1) # Eliminamos "columnas_eliminar" (sobretodo serán claves primarias)
    df_res = df_res.drop_duplicates() # Eliminamos valores repetidos (se generarán varios ahora que hemos quitado claves primarias)
    print("El tamaño del dataframe era: {}, y ahora es: {}.".format(df.shape,df_res.shape))
    return  df_res

# Esta función recibe un dataframe y una variable y devuelve un nuevo dataframe agrupado por la variable var, cuya columnas son count y
# porcentaje. Count hace referencia al número de apariciones de cada valor de la variable var, y porcentaje el porcentaje que supone sobre
# el total de los datos.

def Tabla_var(df,var):
    df_count = pd.DataFrame(df.groupby([var])[var].count().rename('count'))
    df_count['porcentaje'] = df_count['count']/df_count['count'].sum()
    df_count = df_count.sort_values(by="count", ascending=False).reset_index()
    return df_count

# La función Preprocessing recibe un dataframe y transforma la variable severidad por los valores 1 y 0, respectivamente, así como reemplaza
# los valores X, N, U por np.nan. Además, se transforma la variable sexo_pasajero para tenerla como variable numérica.
# En resumidas cuentas, se realiza un preprocessing sobre nuestro dataframe, como bien indica el nombre de la función.

def Preprocesing(df):
    # Reemplazamos los valores de severidad de 2 y 1 por 1 y 0.
    df.replace({'severidad' : {2:1, 1:0}}, inplace=True)
    # Reemplazamos las U, X y N por NAN.
    df = df.replace({"U": np.nan, "UU": np.nan, "UUUU": np.nan, "X": np.nan, "XX": np.nan, "XXXX": np.nan, "N": np.nan, "NN": np.nan, "NNNN": np.nan})
    # Reemplazamos las Q por ceros.
    df = df.replace({"Q": 0, "QQ": 0, "QQQQ": 0})
    df.sexo_pasajero.replace({"M":1, "F":0}, inplace=True)
    # Pasamos a numerico el resto de valores de todas las columnas (costoso computacional)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = 'ignore', downcast= 'integer')
    # Creamos la variable más útil "edad_vehículo", en lugar  de "año_vehículo".
    df['edad_vehiculo']=df['año']-df['año_vehiculo']
    df=df.drop(['año_vehiculo'], axis=1)

    return df

# Esta función recibe un dataframe y devuelve la matriz de correlación entre las variables.

def Correlacion(df):
    plt.figure(figsize=(15,7.5))
    sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=+1, cmap='RdYlGn')

# Esta función nos devuelve la distribución de los valores nulos por columnas o filas (en función de string) y porcentaje 
# nulos/datos_totales.

def Nulos_por(df,string):
    if string == "columnas":
        pd_series_null = df.isnull().sum().sort_values(ascending=False)
    elif string == "filas":
        pd_series_null = df.isnull().sum(axis=1).sort_values(ascending=False)
    else :
        raise ValueError("La cadena de texto introducida debe ser o 'columnas' o 'filas'.")
    
    pd_null = pd.DataFrame(pd_series_null, columns=['Nulos por {}'.format(string)])     
    pd_null['Nulos por {} porcentaje'.format(string)] = pd_null['Nulos por {}'.format(string)]/df.shape[0]*100

    return pd_null

# La función Registros_nulos recibe un dataframe y devuelve la cantidad exacta de observaciones con registros nulos. 

def Registros_nulos(df):
    pd_null_filas = Nulos_por(df,"filas")
    nulos_totales = pd_null_filas.groupby('Nulos por filas').count()
    nulos_totales = nulos_totales.reset_index()
    nulos_totales['cantidad'] = nulos_totales['Nulos por filas porcentaje']
    nulos_totales['porcentaje'] = nulos_totales['cantidad']/(df.shape[0])*100
    nulos_totales = nulos_totales.drop(['Nulos por filas porcentaje'],axis=1)
    return nulos_totales

# La función Eliminar_nulos_filas_n recibe como parámetros un dataframe y un entero, y elimina todas las observaciones de nuestro dataframe
# tal que superan un número n de variables con registro nulo.

def Eliminar_nulos_filas_n(df,n):
    # Añadimos una nueva columna auxiliar que nos muestra la cantidad de atributos nulos que tiene cada registro
    df['Nulos por filas']=df.isnull().sum(axis=1).sort_values(ascending=False)
    # Nos quedamos con los valores nulos menores o iguales que 7
    df = df[df['Nulos por filas']<=n]
    # Eliminamos la columna auxiliar
    df = df.drop(['Nulos por filas'],axis=1)
    return df

# La siguiente función recibe como entrada un dataframe y básicamente imputa aquellos valores missing por el valor más frecuente, es decir, 
# la moda.

def Imputer_moda(df):
    df_no_missing = df
    imputer = SimpleImputer(strategy='most_frequent') # Por la moda
    imputer = imputer.fit(df_no_missing)
    df_no_missing = imputer.transform(df_no_missing)
    df_no_missing = pd.DataFrame(df_no_missing, columns = df.columns)
    return df_no_missing
