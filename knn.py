import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Leukemia_GSE22529_U133A.csv')
#data = pd.read_csv("Pancreatic_GSE16515.csv")
#data = pd.read_csv("Throat_GSE12452.csv")

X = data.drop(['samples', 'type'], axis=1).values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = data['type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracies = []
for k in range(1, 32):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

best_k = np.argmax(accuracies) + 1
print(f"Melhor valor de K: {best_k}")

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo: {accuracy:.2f}")

plt.scatter(range(len(y_test)), y_test, color='blue', label='Real')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Previsto')
plt.xlabel('Amostras')
plt.ylabel('Tipo')
plt.title('Previsões do k-NN')
plt.legend()
plt.show()