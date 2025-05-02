# scripts/build_embeddings.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

produtos = [
    "banana prata", "banana ouro", "banana", "banana maçã",
    "maçã", "batata", "cenoura", "pipoca", "bala de banana"
]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(produtos, batch_size=256, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# Salva vetor e índice
np.save("produtos_embeddings.npy", embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss.index")

# Salva produtos.txt com os nomes, se necessário
with open("produtos.txt", "w") as f:
    f.write("\n".join(produtos))



model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
index = faiss.read_index("faiss.index")
embeddings = np.load("produtos_embeddings.npy")
with open("produtos.txt") as f:
    produtos_base = [l.strip() for l in f.readlines()]

faiss.normalize_L2(embeddings)

produto_search = 'BALA BANANA PRIMOR UN'

emb = model.encode([produto_search])
faiss.normalize_L2(emb)
dist, idx = index.search(emb, 3)

resultados = [
    {
        "produto": produtos_base[i],
        "similaridade": f"{dist[0][n]:.2%}"
    } for n, i in enumerate(idx[0])
]

print( {"entrada": produto_search, "resultados": resultados})