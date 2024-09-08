import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('dataSets/damCombustible_cleaned.csv')

# Convertir 'Odómetro' y 'Cant.' a numérico, reemplazando las comas y valores nulos
df['Odómetro'] = pd.to_numeric(df['Odómetro'], errors='coerce')
df['Cant.'] = df['Cant.'].str.replace(',', '').astype(float)

# Convertir 'Vehículo' a un formato numérico
df['Vehículo'] = df['Vehículo'].astype('category').cat.codes

# Dropeamos columnas que no se usan
df = df.drop(columns=['Nro.', 'Horómetro', 'Fecha', 'Tanqueo Full', 'Costo por Volumen', 'Unidad', 'Costo Total', 'Tipo', 'Unnamed: 11'])

scaler = MinMaxScaler()

# Normalizar las columnas 'Odómetro' y 'Cant.'
df[['Odómetro', 'Cant.']] = scaler.fit_transform(df[['Odómetro', 'Cant.']])

df = df.rename(columns={'Vehículo':'vehicle','Odómetro': 'odometer', 'Cant.': 'quantity'})

# Variables independientes
x = df[['odometer', 'quantity']]

# Variable dependiente
y = df['vehicle']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

