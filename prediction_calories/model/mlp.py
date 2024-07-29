#cirar código do modelo
#usar modelo do vinho

import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf

input_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/input_train.csv')
output_train = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/train/output_train.csv')


model = tf.keras.models.Squential()
model.add(tf.keras.layers.Dense(units=28, activation='relu', input_shape=(input_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=64, activation ='relu'))
model.add(tf.keras.layers.Dense(units=1))
