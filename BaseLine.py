import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib


# Загрузка данных из CSV-файла
df = pd.read_csv('C:/Users/FRXZ/Downloads/rez.csv')

# Определение входных и выходных данных
X = df[['discrepancy', 'T_a', 'U_reg', 'target', 'T_a_in']]  # Входные параметры
y = df['delta_U']  # Целевой параметр

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(4, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

# Оценка модели на тестовых данных
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Прогнозирование
y_pred = model.predict(X_test)

# Сохранение скалера
joblib.dump(scaler, 'scaler.joblib')
# Сохранение модели
model.save('model.h5')

