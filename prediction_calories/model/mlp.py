import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv')

scaler_input = StandardScaler()
i_train = scaler_input.fit_transform(input_train)

scaler_output = StandardScaler()
o_train = scaler_output.fit_transform(output_train.values.reshape(-1, 1))

# Definir o modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train.shape[1],)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='linear'))  # Ajustar unidades conforme a saída

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Compilar o modelo
model.compile(optimizer='adam', loss='mse')

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(i_train, o_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Fazer previsões
predicoes = model.predict(i_train)

# Se necessário, aplicar inversão de normalização
predicoes_inverted = scaler_output.inverse_transform(predicoes)

# Exibir previsões
print(predicoes_inverted)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.yscale('log')
plt.show()
