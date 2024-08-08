import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Carregar os dados
input_train_ = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv').squeeze()
input_test_ = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test.csv')
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv').squeeze()

# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train_)
input_test = scaler_input.transform(input_test_)

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

# print selected feature name column in dataframe 
selected_feature_names = input_train_.columns[selected_features]
print(selected_feature_names)
