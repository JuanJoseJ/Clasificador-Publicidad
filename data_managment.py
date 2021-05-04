import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', None)

# Metodo para traer datos desde un archivo csv en formato DataFrame
def data_retrive(archivo):
    data = pd.read_csv(archivo, sep=";")
    return data

# Se arregla la información para que sea más manipulable 
def pre_processing(archivo, drops, categorical_data, binary_data, scaled_data):

    data = data_retrive(archivo)

    scaler = MinMaxScaler()

    # Primero se eliminan las columnas que no dan información que pueda
    # ayudar con la predicción 
    # print(data)
    if drops != None:
        data.drop(drops, axis=1, inplace=True)

    # Se transforman las variables que representen datos categoricos no numericos
    # y se reintegran en la información que sera utilizada 
    for column in categorical_data:
        dummie = pd.get_dummies(data[column])
        data = data.join(dummie)
        data.drop(column, axis=1, inplace=True)

    # Los datos que pueden ser transformados a binarios se pasan pues pueden
    # ser más fáciles de manipular. (Male = 0; Female = 1)
    for column in binary_data:
        data[column] = data[column].map({'Male':0, 'Female':1})

    # Para los valores que son más o menos grandes, se deben escalar para
    # no desviar los resultados
    for column in scaled_data:
        data[column] = data[column]/data[column].max(axis=0)

    return data
