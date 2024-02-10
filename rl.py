import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('Leukemia_GSE22529_U133A.csv')
#data = pd.read_csv("Pancreatic_GSE16515.csv")
#data = pd.read_csv("Throat_GSE12452.csv")

X = data.drop('type', axis=1)
y = data['type']

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)
print("Erro Absoluto Médio (MAE):", mae)

y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
precisao = accuracy_score(y_test, y_pred_binary)
print("Precisão do modelo:", precisao)

residuos = y_test - y_pred
plt.scatter(range(len(residuos)), residuos, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Amostra')
plt.ylabel('Resíduos')
plt.title('Resíduos da Regressão Linear')
plt.show()