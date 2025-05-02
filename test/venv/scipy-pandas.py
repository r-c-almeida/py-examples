import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine


df = pd.DataFrame({
    'codigo': ['01', '02', '03'],
    'nome': ['Banana Prata', 'Abacaxi', 'Abacate']
})

productName = 'Banana Prt'


vectorizer = CountVectorizer().fit(df['nome'].tolist() + [productName])
vectors = vectorizer.transform(df['nome'].tolist() + [productName]).toarray()

vector_imput = vectors[-1]
vector_base = vectors[:-1]

df['similarity'] = [
    1 - cosine(vector, vector_imput) for vector in vector_base
]

print(df)