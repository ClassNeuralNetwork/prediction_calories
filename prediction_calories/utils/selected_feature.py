import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Carregar os dados
input_train = pd.read_csv('/home/usuario/Documentos/projetos/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('/home/usuario/Documentos/projetos/prediction_calories/prediction_calories/dataset/train/output_train.csv')
input_test = pd.read_csv('/home/usuario/Documentos/projetos/prediction_calories/prediction_calories/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/usuario/Documentos/projetos/prediction_calories/prediction_calories/dataset/test/input_test.csv')



# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train)
input_test = scaler_input.transform(input_test)

# selected best feature with random forest regressor and recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# random forest regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)

# recursive feature elimination
rfe = RFE(estimator=rfr, n_features_to_select=10)

# fit RFE
rfe.fit(input_train, output_train)

# get selected features
selected_features = rfe.support_

# print the selected features 
print(selected_features)

# selected input_train columns dataframe with selected features
input_train = input_train[:, selected_features]
