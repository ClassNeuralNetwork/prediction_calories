import pandas as pd
from imblearn.over_sampling import SMOTE

#carregar os dados de treino
input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv')
sm = SMOTE(k_neighbors=2)

input_train_balanced, output_train_balanced = sm.fit_resample(input_train,output_train)

#salvar dados balanceados
input_train_balanced.to_csv('input_train_balanced.csv',index=False)
output_train_balanced.to_csv('output_train_balanced.csv',index=False)

# criar pasta model
