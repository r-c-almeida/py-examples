import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(BASE_DIR, "housing.csv")
dados = pd.read_csv(caminho)
print(dados.shape)
print(dados.info())
print(dados.head())


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

print(set(dados['ocean_proximity']))

print(dados['ocean_proximity'].value_counts())

print(dados.describe())

print(dados.hist(bins=50, figsize=(20,15)))
dados.hist(bins=50, figsize=(20,15))


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(dados, test_size=0.2, random_state=7)

print(len(df_train), "Treinamento +", len(df_test), "Test")

dados['median_income'].hist()


dados['income_cat'] = np.ceil(dados['median_income'] /1.5)
dados['income_cat'].where(dados['income_cat'] <5, 5.0, inplace=True)

dados['income_cat'] = pd.cut(dados["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print(dados['income_cat'].value_counts())
dados['income_cat'].hist()


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(dados, dados['income_cat']):
    strat_train_set = dados.loc[train_index]
    strat_test_set = dados.loc[train_index]

print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))

print(dados['income_cat'].value_counts() / len(dados))


plt.show()