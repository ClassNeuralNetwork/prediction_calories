import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import shap

# Carregar os dados
input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv')
input_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test.csv')
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv')

# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train)
input_test = scaler_input.transform(input_test)

scaler_output = StandardScaler()
output_train = scaler_output.fit_transform(output_train.values.reshape(-1, 1))
output_test = scaler_output.transform(output_test.values.reshape(-1, 1))

# Definir o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='linear')  # Saída única para regressão
])

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(input_train, output_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Avaliar o modelo no conjunto de teste
test_loss, test_mae = model.evaluate(input_test, output_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Fazer previsões no conjunto de teste
predicoes = model.predict(input_test)

# Inverter a normalização das previsões
predicoes_inverted = scaler_output.inverse_transform(predicoes)
output_test_inverted = scaler_output.inverse_transform(output_test)

# Exibir previsões
print(predicoes_inverted[:10])
print(output_test_inverted[:10])

# Calcular métricas de avaliação
mse = mean_squared_error(output_test_inverted, predicoes_inverted)
mae = mean_absolute_error(output_test_inverted, predicoes_inverted)
r2 = r2_score(output_test_inverted, predicoes_inverted)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# Plotar a perda durante o treinamento
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Comparar valores reais e previstos
plt.scatter(output_test_inverted, predicoes_inverted, alpha=0.5)
plt.plot([output_test_inverted.min(), output_test_inverted.max()], [output_test_inverted.min(), output_test_inverted.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()

background_data = input_train[:1780]  # Usar um subconjunto dos dados de treinamento
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(input_test[:446])  # Usar um subconjunto dos dados de teste

# Plotar o gráfico de resumo
shap.summary_plot(shap_values, input_test[:446])