# PUERTA XOR
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Dataset de entrenamiento para puerta XOR
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y_train = np.array([[0], [1], [1], [0]], dtype="float32")

# Modelo MLP
model = keras.Sequential()
model.add(layers.Dense(2, input_dim=2, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuración del modelo
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
    metrics=[]
)

# Entrenamiento
fit_history = model.fit(x_train, y_train, epochs=50, batch_size=4)

# Curva de pérdida
loss_curve = fit_history.history['loss']

plt.plot(loss_curve, label='Pérdida')
plt.legend(loc='lower left')
plt.title('Resultado del Entrenamiento')
plt.show()

# Recuperamos bias y weights de la capa oculta
weights_HL, biases_HL = model.layers[0].get_weights()

# Recuperamos bias y weights de la capa de salida
weights_OL, biases_OL = model.layers[1].get_weights()

print(weights_HL)
print(biases_HL)
print(weights_OL)
print(biases_OL)

# Predicciones
prediccion = model.predict(x_train)
print(prediccion)
print(x_train)
print(y_train)

# -------------------------------------------------

# CLASIFICACION SUPERVISADA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar dataset real (flores Iris)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar modelo supervisado (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Evaluar
y_pred = model.predict(X_test)
print("Exactitud del modelo:\n", accuracy_score(y_test, y_pred))

# 5. Predicción con nuevos datos
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]
print("Predicción para la flor nueva:\n", iris.target_names[model.predict(nueva_flor)][0])

# -------------------------------------------------

# CLUSTERING (APRENDIZAJE NO SUPERVISADO)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Crear datos artificiales
X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

# 2. Entrenar modelo no supervisado
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. Visualizar resultados
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200)
plt.title("Clustering con KMeans")
plt.show()

# -------------------------------------------------

# RED NEURONAL SIMPLE - BACKPROPAGATION
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. Datos de entrenamiento (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Crear red neuronal
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # capa oculta
    Dense(1, activation='sigmoid')                  # capa de salida
])

# 3. Compilar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entrenar
model.fit(X, y, epochs=500, verbose=0)

# 5. Evaluar predicciones
print("Predicciones XOR:")
print(X, "\n", model.predict(X).round())
