import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


products = ['Banana prata', 'banana Nanica', 'Abacaxi', 'Maçã']
codes = [1, 2, 3, 4]
product_name = 'Abacate'

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(products)

imput_vec = vectorizer.transform([product_name])

knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn.fit(X, codes)


dist, idx = knn.kneighbors(imput_vec, n_neighbors=1)

print(idx)
print(dist)


pred = knn.predict(imput_vec)

print(pred)