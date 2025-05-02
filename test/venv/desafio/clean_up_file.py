import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import seaborn as sns

headers = [
    "UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM",
    "NUM_UC", "QUADRO", "SEQ", "V9001", "V9002", "V9005", "VALOR_SAUDE",
    "V9010", "V9011", "V9012", "V1904", "V1905", "DEFLATOR",
    "V8000_DEFLA", "V1904_DEFLA", "COD_IMPUT_VALOR", "COD_IMPUT_QUANTIDADE",
    "FATOR_ANUALIZACAO", "PESO", "PESO_FINAL", "RENDA_TOTAL", "V9004"
]

colspecs = [
    (0, 2), (2, 6), (6, 7), (7, 16), (16, 18), (18, 19), (19, 21),
    (21, 23), (23, 30), (30, 32), (32, 36), (36, 46), (46, 48), (48, 50),
    (50, 51), (51, 61), (61, 62), (62, 74), (74, 84), (84, 94),
    (94, 95), (95, 96), (96, 98), (98, 112), (112, 126), (126, 136), (136, 141)
]

uf_map = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF"
}

DIRECTORY_BASE = os.path.dirname(os.path.abspath(__file__))
db_path_collective_cost = os.path.join(DIRECTORY_BASE,"POF","DESPESA_COLETIVA.txt")
db_collective_cost = pd.read_fwf(db_path_collective_cost, encoding='latin1', names=headers, colspecs=colspecs, dtype=str)
print(db_collective_cost.head())
db_collective_cost["VALOR_SAUDE"] = db_collective_cost["VALOR_SAUDE"].astype(float) 
db_collective_cost["RENDA_TOTAL"] = db_collective_cost["RENDA_TOTAL"].astype(float)
db_collective_cost["RENDA_TOTAL_ANUAL"] = db_collective_cost["RENDA_TOTAL"].astype(float) * 12
db_collective_cost["UF"] = db_collective_cost["UF"].astype(int)
db_collective_cost["UF_MAP"] = db_collective_cost["UF"].map(uf_map)


#db_collective_cost_insurance = db_collective_cost[db_collective_cost["V9001"]== 3211101]

#print(db_collective_cost_insurance.head())
#print(db_collective_cost_insurance.count())
columns_to_check = ["UF_MAP", "RENDA_TOTAL", "VALOR_SAUDE"]
db_to_check = db_collective_cost[columns_to_check]
db_to_check["VALOR_SAUDE"] = db_to_check["VALOR_SAUDE"].astype(float)

db_to_check = db_to_check[db_to_check["VALOR_SAUDE"] < 10000 ]

print(db_to_check.head())

ymin = db_to_check["RENDA_TOTAL"].quantile(0.01)
ymax = db_to_check["RENDA_TOTAL"].quantile(0.95)

#plt.figure(figsize=(12,6))
sns.boxplot(x="UF_MAP", y="RENDA_TOTAL", data = db_to_check)
plt.ylim(ymin, ymax)
#plt.xticks(rotation=45)
plt.grid(True)
plt.show()

ymin = db_to_check["VALOR_SAUDE"].quantile(0.01)
ymax = db_to_check["VALOR_SAUDE"].quantile(0.95)
sns.boxplot(x="UF_MAP", y="VALOR_SAUDE", data = db_to_check)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.show()

sns.scatterplot(x="RENDA_TOTAL", y="VALOR_SAUDE", data=db_to_check)
plt.show()

for column in columns_to_check:
    plt.figure()
    plot = db_collective_cost[column]
    sns.histplot(plot, bins=10, kde=True)
    plt.show()


correlation_matrix = db_to_check.corr().round(2)
fog, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data=correlation_matrix, annot=True, linewidths=5, ax=ax)
plt.show()