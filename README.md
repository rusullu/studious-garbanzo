# studious-garbanzo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
data = {
    'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
    'temperature': np.random.randint(-10, 10, size=100),
    'energy_consumption': np.random.randint(200, 500, size=100),
}

df = pd.DataFrame(data)

# Средняя цена электроэнергии (руб/кВт·ч), условные данные
df['price_per_kwh'] = np.linspace(4.5, 6.5, len(df)) + np.random.normal(0, 0.1, len(df))
# ---------------------------

# 2. Предобработка данных
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df = df.drop('date', axis=1)

# =========================================================
# МОДЕЛЬ 1: ЭНЕРГОПОТРЕБЛЕНИЕ
# =========================================================

X_energy = df[['temperature', 'day_of_week', 'month']]
y_energy = df['energy_consumption']

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_energy, y_energy, test_size=0.2, random_state=42)

energy_model = LinearRegression()
energy_model.fit(X_train_e, y_train_e)

y_pred_energy = energy_model.predict(X_test_e)

mae_energy = mean_absolute_error(y_test_e, y_pred_energy)
rmse_energy = np.sqrt(mean_squared_error(y_test_e, y_pred_energy))

print("Прогноз энергопотребления:")
print(f"MAE: {mae_energy}")
print(f"RMSE: {rmse_energy}")

# ---------------- ВИЗУАЛИЗАЦИЯ ЭНЕРГОПОТРЕБЛЕНИЯ ----------------

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test_e,
    y_pred_energy,
    label='Предсказанные значения'
)

plt.plot(
    [y_test_e.min(), y_test_e.max()],
    [y_test_e.min(), y_test_e.max()],
    lw=2,
    label='Линия идеального предсказания'
)

plt.title('Реальные vs Предсказанные значения энергопотребления')
plt.xlabel('Реальное энергопотребление (кВт·ч)')
plt.ylabel('Предсказанное энергопотребление (кВт·ч)')
plt.legend()

# Поясняющая плашка
plt.figtext(
    0.5, -0.15,
    'Синие точки — прогноз модели для каждого дня.\n'
    'Линия показывает идеальный случай, когда прогноз совпадает с реальными данными.\n'
    'Чем ближе точки к линии, тем точнее работает модель.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# -------- ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ ОШИБОК ЭНЕРГОПОТРЕБЛЕНИЯ --------

errors_energy = y_test_e - y_pred_energy

plt.figure(figsize=(10, 6))
sns.histplot(errors_energy, kde=True)

plt.title('Распределение ошибок предсказания энергопотребления')
plt.xlabel('Ошибка предсказания (кВт·ч)')
plt.ylabel('Частота')

plt.figtext(
    0.5, -0.15,
    'Столбики показывают, как часто модель ошибается.\n'
    'Линия отображает общее распределение ошибок.\n'
    'Значения около нуля означают точный прогноз.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# =========================================================
# МОДЕЛЬ 2: ЦЕНА ЭЛЕКТРОЭНЕРГИИ
# =========================================================

X_price = df[['temperature', 'day_of_week', 'month']]
y_price = df['price_per_kwh']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)

price_model = LinearRegression()
price_model.fit(X_train_p, y_train_p)

y_pred_price = price_model.predict(X_test_p)

mae_price = mean_absolute_error(y_test_p, y_pred_price)
rmse_price = np.sqrt(mean_squared_error(y_test_p, y_pred_price))

print("\nПрогноз цены электроэнергии:")
print(f"MAE: {mae_price}")
print(f"RMSE: {rmse_price}")

# ---------------- ВИЗУАЛИЗАЦИЯ ЦЕН ----------------

plt.figure(figsize=(10, 6))
plt.scatter(y_test_p, y_pred_price)
plt.plot(
    [y_test_p.min(), y_test_p.max()],
    [y_test_p.min(), y_test_p.max()],
    lw=2
)

plt.title('Реальные vs Предсказанные цены электроэнергии')
plt.xlabel('Реальная цена (руб/кВт·ч)')
plt.ylabel('Предсказанная цена (руб/кВт·ч)')

plt.figtext(
    0.5, -0.15,
    'Каждая точка — прогноз цены на конкретный день.\n'
    'Линия показывает идеальный случай полного совпадения прогноза и реальности.\n'
    'Чем ближе точки к линии, тем точнее модель.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# -------- ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ ОШИБОК ЦЕНЫ --------

errors_price = y_test_p - y_pred_price

plt.figure(figsize=(10, 6))
sns.histplot(errors_price, kde=True)

plt.title('Распределение ошибок предсказания цены электроэнергии')
plt.xlabel('Ошибка предсказания (руб/кВт·ч)')
plt.ylabel('Частота')

plt.figtext(
    0.5, -0.15,
    'Столбики показывают частоту ошибок прогноза цены.\n'
    'Линия — общее распределение ошибок.\n'
    'Если большинство значений около нуля, прогноз считается точным.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# =========================================================
# ЭКОНОМИЧЕСКИЙ РАСЧЁТ
# =========================================================

df['predicted_energy'] = energy_model.predict(X_energy)
df['predicted_price'] = price_model.predict(X_price)
df['predicted_cost'] = df['predicted_energy'] * df['predicted_price']

print("\nЭкономический прогноз (первые 5 строк):")
print(df[['predicted_energy', 'predicted_price', 'predicted_cost']].head())
