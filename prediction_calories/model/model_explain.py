import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import shap

model = tf.keras.models.load_model('model.keras')

input_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test_standard.csv')
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv')

explainer = shap.Explainer(model.predict, input_test)

shap_values = explainer.shap_values(input_test)

shap.summary_plot(shap_values, input_test, plot_type="bar", feature_names=input_test.columns)

