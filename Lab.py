from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# -----------------
# ЗАГРУЗКА ДАННЫХ
# -----------------
# Загружаем тренировочные и валидационные данные
train_set = pd.read_csv('trainingData.csv')
test_set = pd.read_csv('validationData.csv')

# Проверяем наличие данных
print(f"Размер тренировочной выборки: {train_set.shape}")
print(f"Размер тестовой выборки: {test_set.shape}")


# Обработка пропущенных значений
def preprocess_data(df):
    # Преобразуем все колонки в числовой формат
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Удаляем строки с NaN значениями
    df = df.dropna()
    return df


# Обрабатываем обе выборки
train_set = preprocess_data(train_set)
test_set = preprocess_data(test_set)

print(f"После очистки - тренировочная выборка: {train_set.shape}")
print(f"После очистки - тестовая выборка: {test_set.shape}")

# -----------------
# ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# -----------------
target_column = 'LONGITUDE'  # Можно изменить на 'LATITUDE', 'FLOOR', 'BUILDINGID'
print(f"Целевая переменная: {target_column}")

# -----------------
# №1 ПОДГОТОВКА ДАННЫХ
# -----------------
# Разделяем на признаки (Wi-Fi сигналы) и целевую переменную
exclude_columns = [target_column, 'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID',
                   'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']

y_train_set = train_set[target_column]
x_train_set = train_set.drop(columns=[col for col in exclude_columns if col in train_set.columns])

y_test_set = test_set[target_column]
x_test_set = test_set.drop(columns=[col for col in exclude_columns if col in test_set.columns])

print('ПУНКТ 1:\t\tПРОЙДЕН')
print(f"Количество признаков (Wi-Fi точек): {x_train_set.shape[1]}")

# -----------------
# УМЕНЬШЕНИЕ РАЗМЕРНОСТИ
# -----------------
# Отбираем только 50 самых важных признаков чтобы избежать проблем с памятью
selector = SelectKBest(score_func=f_regression, k=min(50, x_train_set.shape[1]))
x_train_reduced = selector.fit_transform(x_train_set, y_train_set)
x_test_reduced = selector.transform(x_test_set)

print(f"После уменьшения размерности: {x_train_reduced.shape[1]} признаков")

# -----------------
# №2 ЛИНЕЙНАЯ РЕГРЕССИЯ (НА УМЕНЬШЕННЫХ ДАННЫХ)
# -----------------
model = LinearRegression()
model.fit(x_train_reduced, y_train_set)

# Предсказания
y_train_pred = model.predict(x_train_reduced)
y_test_pred = model.predict(x_test_reduced)

print('ПУНКТ 2:\t\tПРОЙДЕН')

# -----------------
# №3 ОЦЕНКА КАЧЕСТВА
# -----------------
r2_train = r2_score(y_train_set, y_train_pred)
r2_test = r2_score(y_test_set, y_test_pred)

mse_train = mean_squared_error(y_train_set, y_train_pred)
mse_test = mean_squared_error(y_test_set, y_test_pred)

print(f'R² на тренировочной выборке: {r2_train:.4f}')
print(f'R² на тестовой выборке: {r2_test:.4f}')
print(f'MSE на тренировочной выборке: {mse_train:.4f}')
print(f'MSE на тестовой выборке: {mse_test:.4f}')

print('ПУНКТ 3:\t\tПРОЙДЕН')

# -----------------
# №4 ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (ТОЛЬКО 2-Я СТЕПЕНЬ)
# -----------------
degrees = [1, 2]  # Убираем высшие степени чтобы избежать проблем с памятью

r2_train_list = []
r2_test_list = []

for degree in degrees:
    print(f"Обработка степени {degree}...")

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression),
    ])

    pipeline.fit(x_train_reduced, y_train_set)

    y_train_pred = pipeline.predict(x_train_reduced)
    y_test_pred = pipeline.predict(x_test_reduced)

    r2_train_list.append(r2_score(y_train_set, y_train_pred))
    r2_test_list.append(r2_score(y_test_set, y_test_pred))

# Визуализация результатов
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(degrees, r2_train_list, label="Train R²", marker='o', linewidth=2)
plt.plot(degrees, r2_test_list, label="Test R²", marker='o', linewidth=2)
plt.ylabel("R²")
plt.xlabel("Степень полинома")
plt.title(f"Полиномиальная регрессия\nЦелевая: {target_column}")
plt.legend()
plt.grid(True)

# Находим оптимальную степень
optimal_degree_index = np.argmax(r2_test_list)
optimal_degree = degrees[optimal_degree_index]
print(f'Оптимальная степень полинома: {optimal_degree}')

print('ПУНКТ 4:\t\tПРОЙДЕН')

# -----------------
# №5 RIDGE РЕГРЕССИЯ
# -----------------
alpha = np.logspace(-4, 3, 8)  # Уменьшаем количество альфа для скорости

r2_train_ridge = []
r2_test_ridge = []

for a in alpha:
    print(f"Обработка alpha={a:.6f}...")

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=optimal_degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a, max_iter=1000))
    ])

    pipeline.fit(x_train_reduced, y_train_set)

    y_train_pred = pipeline.predict(x_train_reduced)
    y_test_pred = pipeline.predict(x_test_reduced)

    r2_train_ridge.append(r2_score(y_train_set, y_train_pred))
    r2_test_ridge.append(r2_score(y_test_set, y_test_pred))

# Визуализация Ridge регрессии
plt.subplot(1, 2, 2)
plt.semilogx(alpha, r2_train_ridge, label="Train R²", marker='o', markersize=5, linewidth=2)
plt.semilogx(alpha, r2_test_ridge, label="Test R²", marker='o', markersize=5, linewidth=2)
plt.xlabel("Альфа (логарифмическая шкала)")
plt.ylabel("R²")
plt.title(f"Ridge регрессия\nСтепень: {optimal_degree}")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Находим оптимальное alpha
optimal_alpha_index = np.argmax(r2_test_ridge)
optimal_alpha = alpha[optimal_alpha_index]
print(f'Оптимальное alpha для Ridge: {optimal_alpha:.6f}')

# Финальная модель с оптимальными параметрами
final_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=optimal_degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=optimal_alpha, max_iter=1000))
])

final_pipeline.fit(x_train_reduced, y_train_set)

y_train_final = final_pipeline.predict(x_train_reduced)
y_test_final = final_pipeline.predict(x_test_reduced)

final_r2_train = r2_score(y_train_set, y_train_final)
final_r2_test = r2_score(y_test_set, y_test_final)

print('\n' + '=' * 60)
print('ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:')
print(f'Целевая переменная: {target_column}')
print(f'R² на тренировочной выборке: {final_r2_train:.4f}')
print(f'R² на тестовой выборке: {final_r2_test:.4f}')
print(f'Оптимальная степень полинома: {optimal_degree}')
print(f'Оптимальный параметр регуляризации: {optimal_alpha:.6f}')
print('=' * 60)

print('ПУНКТ 5:\t\tПРОЙДЕН')
print('\nВСЕ ЭТАПЫ ЗАВЕРШЕНЫ!')

# Дополнительно: важность признаков
print(f"\nТоп-10 самых важных Wi-Fi точек для предсказания {target_column}:")
feature_scores = selector.scores_
important_features_idx = np.argsort(feature_scores)[-10:][::-1]
important_features = x_train_set.columns[important_features_idx]

for i, (feature, score) in enumerate(zip(important_features, feature_scores[important_features_idx])):
    print(f"{i + 1}. {feature}: {score:.2f}")