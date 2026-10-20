import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Фиксируем случайные значения, чтобы результаты повторялись при запуске
np.random.seed(42)

# 1. Создание искусственных данных
# Это пример работы модели: данные не взяты из реальной статистики.
data = {
    'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
    'temperature': np.random.randint(-10, 10, size=100),
    'energy_consumption': np.random.randint(200, 500, size=100),
}

df = pd.DataFrame(data)

# Условная цена электроэнергии (руб/кВт·ч), также искусственные данные
df['price_per_kwh'] = (
    np.linspace(4.5, 6.5, len(df))
    + np.random.normal(0, 0.1, len(df))
)

# 2. Предобработка данных: извлекаем день недели и месяц
# Для модели используем эти признаки вместо исходной даты.
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df = df.drop('date', axis=1)

# =========================================================
# МОДЕЛЬ 1: ЭНЕРГОПОТРЕБЛЕНИЕ
# =========================================================

# Признаки, по которым модель делает прогноз, и целевое значение
X_energy = df[['temperature', 'day_of_week', 'month']]
y_energy = df['energy_consumption']

# Делим данные: 80% для обучения, 20% для проверки
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_energy, y_energy, test_size=0.2, random_state=42
)

# Создаём модель и обучаем её на тренировочных данных
energy_model = LinearRegression()
energy_model.fit(X_train_e, y_train_e)

# Проверяем модель на данных, которые она не видела при обучении
y_pred_energy = energy_model.predict(X_test_e)

# Оцениваем ошибки прогноза
mae_energy = mean_absolute_error(y_test_e, y_pred_energy)
rmse_energy = np.sqrt(mean_squared_error(y_test_e, y_pred_energy))

print('Прогноз энергопотребления (искусственные данные):')
print(f'Средняя абсолютная ошибка (MAE): {mae_energy:.2f} кВт·ч')
print(f'Корень из средней квадратичной ошибки (RMSE): {rmse_energy:.2f} кВт·ч')

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

plt.title('Реальные и предсказанные значения энергопотребления')
plt.xlabel('Значение в искусственных данных (кВт·ч)')
plt.ylabel('Предсказанное энергопотребление (кВт·ч)')
plt.legend()

# Краткое объяснение графика
plt.figtext(
    0.5, -0.15,
    'Каждая точка — прогноз для одного дня.\n'
    'Линия показывает идеальный случай, когда прогноз совпадает с заданным значением.\n'
    'Чем ближе точки к линии, тем точнее прогноз.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# -------- ВИЗУАЛИЗАЦИЯ ОШИБОК ЭНЕРГОПОТРЕБЛЕНИЯ --------

errors_energy = y_test_e - y_pred_energy

plt.figure(figsize=(10, 6))
sns.histplot(errors_energy, kde=True)

plt.title('Распределение ошибок прогноза энергопотребления')
plt.xlabel('Ошибка прогноза (кВт·ч)')
plt.ylabel('Частота')

plt.figtext(
    0.5, -0.15,
    'Столбики показывают, как часто встречаются ошибки разного размера.\n'
    'Значения около нуля означают, что прогноз близок к заданному значению.',
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

# Делим данные на обучающую и проверочную части
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)

# Создаём и обучаем модель для прогноза цены
price_model = LinearRegression()
price_model.fit(X_train_p, y_train_p)

# Проверяем прогноз на данных, не использованных при обучении
y_pred_price = price_model.predict(X_test_p)

# Оцениваем ошибки прогноза цены
mae_price = mean_absolute_error(y_test_p, y_pred_price)
rmse_price = np.sqrt(mean_squared_error(y_test_p, y_pred_price))

print('\nПрогноз цены электроэнергии (искусственные данные):')
print(f'Средняя абсолютная ошибка (MAE): {mae_price:.2f} руб/кВт·ч')
print(f'Корень из средней квадратичной ошибки (RMSE): {rmse_price:.2f} руб/кВт·ч')

# ---------------- ВИЗУАЛИЗАЦИЯ ЦЕН ----------------

plt.figure(figsize=(10, 6))
plt.scatter(y_test_p, y_pred_price, label='Предсказанные значения')
plt.plot(
    [y_test_p.min(), y_test_p.max()],
    [y_test_p.min(), y_test_p.max()],
    lw=2,
    label='Линия идеального предсказания'
)

plt.title('Заданные и предсказанные цены электроэнергии')
plt.xlabel('Цена в искусственных данных (руб/кВт·ч)')
plt.ylabel('Предсказанная цена (руб/кВт·ч)')
plt.legend()

plt.figtext(
    0.5, -0.15,
    'Каждая точка — прогноз цены для одного дня.\n'
    'Линия показывает идеальное совпадение прогноза с заданным значением.\n'
    'Чем ближе точки к линии, тем точнее прогноз.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# -------- ВИЗУАЛИЗАЦИЯ ОШИБОК ПРОГНОЗА ЦЕНЫ --------

errors_price = y_test_p - y_pred_price

plt.figure(figsize=(10, 6))
sns.histplot(errors_price, kde=True)

plt.title('Распределение ошибок прогноза цены электроэнергии')
plt.xlabel('Ошибка прогноза (руб/кВт·ч)')
plt.ylabel('Частота')

plt.figtext(
    0.5, -0.15,
    'Столбики показывают частоту ошибок разного размера.\n'
    'Значения около нуля означают, что прогноз близок к заданной цене.',
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.show()

# =========================================================
# ЭКОНОМИЧЕСКИЙ РАСЧЁТ
# =========================================================

# Строим прогнозы для всех строк искусственного набора данных
df['predicted_energy'] = energy_model.predict(X_energy)
df['predicted_price'] = price_model.predict(X_price)

# Умножаем прогноз энергопотребления на прогноз цены
df['predicted_cost'] = (
    df['predicted_energy'] * df['predicted_price']
)

print('\nЭкономический расчёт по прогнозам (первые 5 строк):')
print(df[['predicted_energy', 'predicted_price', 'predicted_cost']].head().round(2))
print('\nВажно: все исходные данные искусственные. Результаты показывают принцип работы модели,')
print('а не точный прогноз реального энергопотребления или стоимости электроэнергии.')
