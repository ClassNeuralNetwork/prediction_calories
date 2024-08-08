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

# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train)
input_test = scaler_input.transform(input_test)

pd.DataFrame(input_train).to_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train_standard.csv', index=False)
pd.DataFrame(input_test).to_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test_standard.csv', index=False)

# Definir o modelo ajustado
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01), input_shape=(input_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear') #saída única para regressão
])

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(input_train, output_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Salvar History
pd.DataFrame(history.history).to_csv('loss.csv', index=False)

# Salvar o modelo
model.save('model.keras')
