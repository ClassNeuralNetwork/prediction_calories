import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Carregar os dados
input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv')
input_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test.csv')
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv')

# selected feature 
selected_feature = ['water_g', 'lipid_g', 'ash_g', 'carbohydrates_g', 'fiber_g',
       'calcium_mg', 'potassium_mg', 'sodium_mg', 'saturated_fat_g',
       'monounsaturated_fat_g']

input_train = input_train[selected_feature]
input_test = input_test[selected_feature]

# Normalizar os dados
scaler_input = StandardScaler()
input_train_scaled = scaler_input.fit_transform(input_train)
input_test_scaled = scaler_input.transform(input_test)

# Criar DataFrame com os nomes das colunas
input_train_standard = pd.DataFrame(input_train_scaled, columns=selected_feature)
input_test_standard = pd.DataFrame(input_test_scaled, columns=selected_feature)

# Salvar os DataFrames como CSV
input_train_standard.to_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train_standard.csv', index=False)
input_test_standard.to_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test_standard.csv', index=False)

# Definir o modelo ajustado
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.1), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear') #saída única para regressão
])

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(input_train_scaled, output_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Salvar History
pd.DataFrame(history.history).to_csv('loss.csv', index=False)

# Salvar o modelo
model.save('model.keras')
