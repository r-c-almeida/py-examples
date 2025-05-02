import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(BASE_DIR, "Sorvete.xlsx")
dados = pd.read_excel(caminho)
print(dados.head())


plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
plt.xlabel("Temperatura (C)")
plt.ylabel("Vendas (Milhares)")
plt.title("Relação entre vendas e temperatura")
plt.show()

print(dados.corr())


x = dados[['Temperatura']]
y = dados[['Vendas_Sorvetes']]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

erro_medio_quadratico = mean_squared_error(y_test, previsoes)
erro_absoluto_medio = mean_absolute_error(y_test, previsoes)
r_quadrado = r2_score(y_test,previsoes)

print(f'Erro Quadratico: {erro_medio_quadratico}')
print(f'Erro Absoluto: {erro_absoluto_medio}')
print(f'R2 *Coeficiente de determinação {r_quadrado}')


plt.scatter(X_test, y_test, label = 'REAL')
plt.scatter(X_test, previsoes, label = 'Previsto', color='red')
plt.xlabel('Temperatura')
plt.ylabel('Quantidade Vendas')
plt.title('Previsões do modelo')
plt.legend()
plt.show()