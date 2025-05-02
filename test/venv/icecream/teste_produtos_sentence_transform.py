from sentence_transformers import SentenceTransformer, util
import torch

# Base de produtos
produtos_base = [
    "banana prata", "banana ouro", "banana", "banana maçã",
    "maçã", "batata", "cenoura", "pipoca"
]

# Produto recebido
entrada = "banana maçã 400gr"

# Carrega modelo leve e eficiente
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Codifica base e entrada
embeddings_base = model.encode(produtos_base, convert_to_tensor=True)
embedding_entrada = model.encode(entrada, convert_to_tensor=True)

# Calcula similaridade
cosine_scores = util.pytorch_cos_sim(embedding_entrada, embeddings_base).squeeze()

# Encontra o mais similar
indice_mais_similar = torch.argmax(cosine_scores)
produto_encontrado = produtos_base[indice_mais_similar]
score = cosine_scores[indice_mais_similar].item()

print(f"Produto mais similar: {produto_encontrado}")
print(f"Similaridade: {score:.2%}")
