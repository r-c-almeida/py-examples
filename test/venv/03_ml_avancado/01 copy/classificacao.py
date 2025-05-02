import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(BASE_DIR, "gaf_esp.xlsx")
dados = pd.read_excel(caminho)

vectorizer = TfidfVectorizer(ngram_range=(2,3))
x = vectorizer.fit_transform(dados['Parceiro'])

print(dados.head())
print(dados.describe())

#dados.plot.scatter(x='Item', y='Parceiro')
#plt.show()

#x = dados[['Item', 'Parceiro']]
y = dados['Item']

modelo_classificador = KNeighborsClassifier(n_neighbors=3)
modelo_classificador.fit(x, y)

def prever_item(descricao_parceiro, threashold: float = 0.4):
    desc_parceiro_vet = vectorizer.transform([descricao_parceiro])
    distancias, indice = modelo_classificador.kneighbors(desc_parceiro_vet, n_neighbors=3)
    
    for i in range(3):
        item = y.iloc[indice[0][i]]
        distancia = distancias[0][i]
        print({"item_predito": item, "distancia": distancia})



    #if distancia < threashold:
    #    return {"item_predito": item, "distancia": distancia}
    #else:
    #    return {"item_predito": None, "distancia": distancia}



exemplo = "FRUTA BANANA NANICA KG"
print(prever_item(exemplo))