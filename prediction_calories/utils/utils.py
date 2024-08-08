import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Carregar dataset
dt = pd.read_csv('C:/project_topicos_especiais/prediction_calories/prediction_calories/dataset/ABBREV.csv')

new_column_names = {
    'NDB_No': 'ndb_no',
    'Shrt_Desc': 'short_desc',
    'Water_(g)': 'water_g',
    'Energ_Kcal': 'energy_kcal',
    'Protein_(g)': 'protein_g',
    'Lipid_Tot_(g)': 'lipid_g',
    'Ash_(g)': 'ash_g',
    'Carbohydrt_(g)': 'carbohydrates_g',
    'Fiber_TD_(g)': 'fiber_g',
    'Sugar_Tot_(g)': 'sugar_g',
    'Calcium_(mg)': 'calcium_mg',
    'Iron_(mg)': 'iron_mg',
    'Magnesium_(mg)': 'magnesium_mg',
    'Phosphorus_(mg)': 'phosphorus_mg',
    'Potassium_(mg)': 'potassium_mg',
    'Sodium_(mg)': 'sodium_mg',
    'Zinc_(mg)': 'zinc_mg',
    'Copper_mg)': 'copper_mg',
    'Manganese_(mg)': 'manganese_mg',
    'Selenium_(µg)': 'selenium_mcg',
    'Vit_C_(mg)': 'vitamin_c_mg',
    'Thiamin_(mg)': 'thiamin_mg',
    'Riboflavin_(mg)': 'riboflavin_mg',
    'Niacin_(mg)': 'niacin_mg',
    'Panto_Acid_mg)': 'pantothenic_acid_mg',
    'Vit_B6_(mg)': 'vitamin_b6_mg',
    'Folate_Tot_(µg)': 'folate_total_mcg',
    'Folic_Acid_(µg)': 'folic_acid_mcg',
    'Food_Folate_(µg)': 'food_folate_mcg',
    'Folate_DFE_(µg)': 'folate_dfe_mcg',
    'Choline_Tot_ (mg)': 'choline_mg',
    'Vit_B12_(µg)': 'vitamin_b12_mcg',
    'Vit_A_IU': 'vitamin_a_iu',
    'Vit_A_RAE': 'vitamin_a_rae',
    'Retinol_(µg)': 'retinol_mcg',
    'Alpha_Carot_(µg)': 'alpha_carotene_mcg',
    'Beta_Carot_(µg)': 'beta_carotene_mcg',
    'Beta_Crypt_(µg)': 'beta_cryptoxanthin_mcg',
    'Lycopene_(µg)': 'lycopene_mcg',
    'Lut+Zea_ (µg)': 'lutein_zeaxanthin_mcg',
    'Vit_E_(mg)': 'vitamin_e_mg',
    'Vit_D_µg': 'vitamin_d_mcg',
    'Vit_D_IU': 'vitamin_d_iu',
    'Vit_K_(µg)': 'vitamin_k_mcg',
    'FA_Sat_(g)': 'saturated_fat_g',
    'FA_Mono_(g)': 'monounsaturated_fat_g',
    'FA_Poly_(g)': 'polyunsaturated_fat_g',
    'Cholestrl_(mg)': 'cholesterol_mg',
    'GmWt_1': 'serving_1_weight_g',
    'GmWt_Desc1': 'serving_1_desc',
    'GmWt_2': 'serving_2_weight_g',
    'GmWt_Desc2': 'serving_2_desc',
    'Refuse_Pct': 'inedible_percent'
}
dt.rename(columns=new_column_names, inplace=True)

# Verificar colunas e valores ausentes
print(dt.head())

# Colunas com valores object (Shrt_Desc, GmWt_Desc1, GmWt_Desc2)
cols_to_transform = ['Shrt_Desc', 'GmWt_Desc1', 'GmWt_Desc2']

# Transformar colunas categóricas em códigos numéricos
for col in cols_to_transform:
    dt[col] = dt[col].astype('category').cat.codes

# Verificar a transformação
print(dt.info())
dt = dt.dropna()

# Definir a coluna alvo
target_column = 'Energ_Kcal'

# Divisão dos dados
input_train, input_test, output_train, output_test = train_test_split(
    dt.drop(columns=[target_column]), dt[target_column], test_size=0.2, random_state=42)

# Salvar os dados divididos em arquivos CSV
pd.DataFrame(input_train).to_csv('input_train.csv', index=False)
pd.DataFrame(input_test).to_csv('input_test.csv', index=False)
pd.DataFrame(output_train).to_csv('output_train.csv', index=False)
pd.DataFrame(output_test).to_csv('output_test.csv', index=False)
