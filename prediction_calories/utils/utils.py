import tensorflow as tf  # para implementar a rede neural
import numpy as np  # para trabalhar com vetores e math
import pandas as pd  # para carregar e manipular dados
import matplotlib.pyplot as plt  # para gerar gráficos
from sklearn.model_selection import train_test_split  # para dividir o conjunto de dados
from sklearn.preprocessing import StandardScaler  # para padronizar os dados
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Carregar dataset
dt = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/ABBREV.csv')

# Verificar colunas e valores ausentes
print(dt.head())


# Colunas com valores object (Shrt_Desc, GmWt_Desc1, GmWt_Desc2)
cols_to_transform = ['Shrt_Desc', 'GmWt_Desc1', 'GmWt_Desc2']

# Transformar colunas categóricas em códigos numéricos
for col in cols_to_transform:
    dt[col] = dt[col].astype('category').cat.codes

# Verificar a transformação
print(dt.info())
dt = dt.dropna()
# Divisão dos dados
input_train, input_test, output_train, output_test = train_test_split(
    dt.iloc[:, :-1], dt['Energ_Kcal'], test_size=0.2, random_state=42)

# Salvar os dados divididos em arquivos CSV
pd.DataFrame(input_train).to_csv('input_train.csv', index=False)
pd.DataFrame(input_test).to_csv('input_test.csv', index=False)
pd.DataFrame(output_train).to_csv('output_train.csv', index=False)
pd.DataFrame(output_test).to_csv('output_test.csv', index=False)