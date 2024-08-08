import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

#Carregando modelo
model = tf.keras.models.load_model('model.keras')

input_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/input_test_standard.csv')
output_test = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/test/output_test.csv')

#carregando loss
history = pd.read_csv('loss.csv')

# Avaliar o modelo no conjunto de teste
test_loss, test_mae = model.evaluate(input_test, output_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Fazer previsões no conjunto de teste
predicoes = model.predict(input_test)

# Calcular métricas de avaliação
mse = mean_squared_error(output_test, predicoes)
mae = mean_absolute_error(output_test, predicoes)
r2 = r2_score(output_test, predicoes)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# Plotar a perda durante o treinamento
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.legend()
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()


# Comparar valores reais e previstos
plt.scatter(output_test, predicoes, alpha=0.5)
plt.plot([output_test.min(), output_test.max()], [output_test.min(), output_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()
