import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

products = ["Banana prata", "Banana maçã", "Banana nanica", "Abacaxi", "Laranja"]
#productName = "Banana prt uni 10"
productName = 'Banana banana Banana PRATA Amarela Madura 12 Unidades'

vectorizer = CountVectorizer().fit(products+[productName])
vector = vectorizer.transform(products+ [productName]).toarray()

for i, product in enumerate(products):
    yes = 1 - cosine(vector[i], vector[-1])
    print(f'{productName} -> Similarity: {yes:.2f}')