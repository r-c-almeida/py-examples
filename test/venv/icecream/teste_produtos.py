from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Base de produtos
produtos_base = [
    "banana prata", "banana ouro", "banana", "banana maçã",
    "maçã", "batata", "cenoura", "pipoca"
]

# Produto recebido pela API
entrada = "banana maçã 400gr"

# Pré-processamento simples (pode melhorar depois)
def preprocess(text):
    return text.lower().replace("gr", "").replace("ml", "").strip()

produtos_processados = [preprocess(p) for p in produtos_base]
entrada_processada = preprocess(entrada)

# Vetorização
vectorizer = TfidfVectorizer()
vetores = vectorizer.fit_transform(produtos_processados + [entrada_processada])

# Cálculo de similaridade
similaridades = cosine_similarity(vetores[-1], vetores[:-1]).flatten()

# Encontrar o produto mais similar
indice_mais_similar = np.argmax(similaridades)
produto_mais_similar = produtos_base[indice_mais_similar]
score = similaridades[indice_mais_similar]

print(f"Mais parecido: {produto_mais_similar} ({score:.2%} de similaridade)")
