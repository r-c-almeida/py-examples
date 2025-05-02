import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(BASE_DIR, "gaf_esp.xlsx")
dados = pd.read_excel(caminho)

print(dados.head())
print(dados.describe())

dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas')
plt.show()

x = dados[['Comprimento do Abdômen', 'Comprimento das Antenas']]
y = dados['Espécie']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print(list(y_train).count('Gafanhoto'))
print(list(y_train).count('Esperança'))

print('Total Base de treino: ', len(x_train))
print('Total Base de teste: ', len(x_test))

modelo_classificador = KNeighborsClassifier(n_neighbors=3)
modelo_classificador.fit(x_train, y_train)

y_predit =  modelo_classificador.predict([[0.55, 4]])
y_predito =  modelo_classificador.predict(x_test)

print(y_predit)
print(accuracy_score(y_true = y_test, y_pred = y_predito))
