import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import shap

# Carregar o modelo
model = tf.keras.models.load_model('C:/project_topicos_especiais/prediction_calories/prediction_calories/model/model.keras')

# Definir as features selecionadas
selected_feature = ['water_g', 'lipid_g', 'ash_g', 'carbohydrates_g',
                    'fiber_g', 'calcium_mg', 'potassium_mg', 'sodium_mg',
                    'saturated_fat_g', 'monounsaturated_fat_g']

# Carregar os dados com as features selecionadas
input_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test_standard.csv', usecols=selected_feature)
input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train_standard.csv', usecols=selected_feature)
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv')

# Configurar o SHAP explainer
explainer = shap.KernelExplainer(model.predict, input_train)

# Calcular os SHAP values
shap_values = explainer.shap_values(input_test)

# Plotar o gráfico de resumo
shap.summary_plot(shap_values, input_test, plot_type="bar", feature_names=selected_feature)

# Plotar o gráfico de dependência para a primeira feature
shap.dependence_plot(selected_feature[0], shap_values, input_test)
