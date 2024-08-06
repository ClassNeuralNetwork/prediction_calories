import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Definir a coluna alvo
target_column = 'Energ_Kcal'

# Divisão dos dados
input_train, input_test, output_train, output_test = train_test_split(
    dt.drop(columns=[target_column]), dt[target_column], test_size=0.2, random_state=42)

# Salvar os dados divididos em arquivos CSV
pd.DataFrame(input_train).to_csv('input_train.csv', index=False)
pd.DataFrame(input_test).to_csv('input_test.csv', index=False)
pd.DataFrame(output_train).to_csv('output_train.csv', index=False)
pd.DataFrame(output_test).to_csv('output_test.csv', index=False)
