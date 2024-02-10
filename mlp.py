import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('Leukemia_GSE22529_U133A.csv')
#data = pd.read_csv("Pancreatic_GSE16515.csv")
#data = pd.read_csv("Throat_GSE12452.csv")

X = data.drop("type", axis=1)
y = data["type"]

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(48, 48), max_iter=1000, learning_rate_init=0.05, activation="relu", random_state=1)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

precisao = mlp.score(X_test, y_test)
print("Precisão:", precisao)

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, label='Rótulos Verdadeiros', color='blue', s=50)
plt.scatter(range(len(y_test)), y_pred, label='Rótulos Previstos', color='red', s=50)

plt.title("Comparação entre Rótulos Verdadeiros e Previstos")
plt.xlabel("Amostra")
plt.ylabel("Rótulo")
plt.legend()
plt.show()