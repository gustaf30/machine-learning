import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Leukemia_GSE22529_U133A.csv')
#data = pd.read_csv("Pancreatic_GSE16515.csv")
#data = pd.read_csv("Throat_GSE12452.csv")

features = data.drop('type', axis=1)
target = data['type']

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

dot_data = export_graphviz(model, out_file=None, feature_names=features.columns, class_names=target.unique(), filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('tree.png')