from rapidfuzz import process, fuzz

# Base de produtos
produtos_base = [
    "banana prata", "banana ouro", "banana", "banana maçã",
    "maçã", "batata", "cenoura", "pipoca"
]

# Produto recebido
entrada = "banana maçã 400gr"

# Match utilizando token sort ratio (ignora ordem e tokens extras)
resultado = process.extractOne(
    query=entrada,
    choices=produtos_base,
    scorer=fuzz.token_sort_ratio
)

produto_encontrado, score, _ = resultado

print(f"Produto mais similar: {produto_encontrado}")
print(f"Similaridade: {score:.2f}%")
