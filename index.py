from data_managment import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt


archivo = "PublicidadRedesSociales.csv"
archivo_test = "pronosticoNuevos_RS.csv"

drops = ["User ID"]  # Dropeo esto porque no hace falta el ID
cat_data = []   # Datos categoricos que deben ser transformados a
                # cuantitativos para ser más fácilmente manejados
bin_data = ["Gender"]   # Datos que deben ser transformados a binario porque es más
                        # facil de manejar que como strings o bool
target_var = "Purchased"
scal_data = ['Age', 'EstimatedSalary']

datos = pre_processing(archivo, drops, cat_data, bin_data, scal_data)


# Separo las variables a evaluar y la que representa un 'target'
# y = target; x = explanatory
y = datos[target_var].copy()
x = datos.drop([target_var], axis=1)

# Divido los datos entre test y entrenamiento para ser utilizados luego
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Almaceno los reusltados de la regresión logística y redes neuronales
modelo_logistico = LogisticRegression(random_state=1)
modelo_NN = MLPClassifier(max_iter=1000, random_state=1)
modelo_logistico.fit(x_train, y_train)
modelo_NN.fit(x_train, y_train)

# Calculo la predicción de resultados usando el test data
y_pred_log = pd.Series(modelo_logistico.predict(x_test))
y_pred_NN = pd.Series(modelo_NN.predict(x_test))
y_test = y_test.reset_index(drop=True)
z_log = pd.concat([y_test,y_pred_log], axis=1)
z_NN = pd.concat([y_test,y_pred_NN], axis=1)
z_log.columns = ['True', 'Prediction']
z_NN.columns = ['True', 'Prediction']

print("Resultados Reg. Logística")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_log))
print("Precision:", metrics.precision_score(y_test, y_pred_log))
print("Recall:", metrics.recall_score(y_test, y_pred_log))
print("--------------------------------------------------------------------------")
print("Resultados Redes Neuronales")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_NN))
print("Precision:", metrics.precision_score(y_test, y_pred_NN))
print("Recall:", metrics.recall_score(y_test, y_pred_NN))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_log)

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

sns.set()
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True')
plt.xlabel('Predicted')

plt.show()

# CON ARCHIVO NUEVO
datos_test = pre_processing(archivo_test, drops, cat_data,bin_data,scal_data)

# Separo las variables a evaluar y la que representa un 'target'
# y = target; x = explanatory
x_test2 = datos_test.drop([target_var], axis=1)

# Calculo la predicción de resultados usando el test data
y_pred2_log = pd.Series(modelo_logistico.predict(x_test2))
y_pred2_NN = pd.Series(modelo_NN.predict(x_test2))

# Uno los valores a la tabla original
x_test2["Predictions_log"] = y_pred2_log
x_test2["Predictions_NN"] = y_pred2_NN

print("----------------------------------------------------------------------------")
print(x_test2)
