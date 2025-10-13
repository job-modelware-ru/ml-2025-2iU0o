import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Загрузка датасета (Wine: химические характеристики вин, 13 признаков, 3 класса)
wine = load_wine()
X = wine.data
y = wine.target
df = pd.DataFrame(X, columns=wine.feature_names)

def manual_min_max_scaler():
    class ManualMinMaxScaler:
        def fit_transform(self, X):
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

        def transform(self, X):
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

    return ManualMinMaxScaler()


def manual_standard_scaler():
    class ManualStandardScaler:
        def fit_transform(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            return (X - self.mean_) / (self.std_ + 1e-8)

        def transform(self, X):
            return (X - self.mean_) / (self.std_ + 1e-8)

    return ManualStandardScaler()


def manual_robust_scaler():
    class ManualRobustScaler:
        def fit_transform(self, X):
            self.median_ = np.median(X, axis=0)
            self.q1_ = np.percentile(X, 25, axis=0)
            self.q3_ = np.percentile(X, 75, axis=0)
            return (X - self.median_) / (self.q3_ - self.q1_ + 1e-8)

        def transform(self, X):
            return (X - self.median_) / (self.q3_ - self.q1_ + 1e-8)

    return ManualRobustScaler()


def manual_log_scaler():
    class ManualLogScaler:
        def fit_transform(self, X):
            self.min_shift = np.min(X, axis=0)
            return np.log1p(X)

        def transform(self, X):
            return np.log1p(X)

    return ManualLogScaler()

scalers = {
    'Без нормализации': None,
    'Min-Max': manual_min_max_scaler(),
    'Z-score': manual_standard_scaler(),
    'Robust': manual_robust_scaler(),
    'Log': manual_log_scaler()
}

models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=10000, solver='saga', tol=1e-4)
}

# Усреднение по 10 запускам
num_runs = 10
all_results = []

for run in range(num_runs):
    print(f"Запуск {run + 1}/{num_runs}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for scaler_name, scaler in scalers.items():
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        for model_name, model in models.items():
            print(f"  Обучение: {scaler_name}, {model_name}")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            all_results.append([run, scaler_name, model_name, acc, f1])

results_df = pd.DataFrame(all_results, columns=['Run', 'Нормализация', 'Модель', 'Accuracy', 'F1-score'])
mean_results = results_df.groupby(['Нормализация', 'Модель']).mean()[['Accuracy', 'F1-score']].reset_index()
print("Усредненные результаты:")
print(mean_results)

mean_results.to_csv('results.csv', index=False)

# Визуализация: гистограмма для proline с Log нормализацией
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['proline'], bins=20, color='blue', alpha=0.7)
plt.title('До нормализации (proline)')
plt.xlabel('Значения (оригинальный диапазон)')
plt.ylabel('Частота')
plt.text(0.05, 0.95, f'Min: {df["proline"].min()}\nMax: {df["proline"].max()}', transform=plt.gca().transAxes, va='top')

scaler = manual_log_scaler()
plt.subplot(1, 2, 2)
plt.hist(scaler.fit_transform(df[['proline']]), bins=20, color='green', alpha=0.7)
plt.title('После Log нормализации (proline)')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('hist_proline_log.png')
plt.show()

# Визуализация: гистограмма для proline с Min-Max нормализацией
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['proline'], bins=20, color='blue', alpha=0.7)
plt.title('До нормализации (proline)')
plt.xlabel('Значения (оригинальный диапазон)')
plt.ylabel('Частота')
plt.text(0.05, 0.95, f'Min: {df["proline"].min()}\nMax: {df["proline"].max()}', transform=plt.gca().transAxes, va='top')

scaler = manual_min_max_scaler()
plt.subplot(1, 2, 2)
plt.hist(scaler.fit_transform(df[['proline']]), bins=20, color='green', alpha=0.7)
plt.title('После Min-Max нормализации (proline)')
plt.xlabel('Значения (в [0,1])')
plt.ylabel('Частота')
plt.text(0.05, 0.95, f'Min: 0\nMax: 1', transform=plt.gca().transAxes, va='top')
plt.tight_layout()
plt.savefig('hist_proline_min_max.png')
plt.show()

# Boxplot для всех признаков Min-Max
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
df.boxplot(ax=ax[0])
ax[0].set_title('До нормализации (все признаки)')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')

scaler_all = manual_min_max_scaler()
df_scaled = pd.DataFrame(scaler_all.fit_transform(df.values), columns=df.columns)
df_scaled.boxplot(ax=ax[1])
ax[1].set_title('После Min-Max нормализации (все признаки)')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
ax[1].set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig('boxplot_min_max.png')
plt.show()

# Графики результатов
acc_pivot = mean_results.pivot(index='Нормализация', columns='Модель', values='Accuracy')
f1_pivot = mean_results.pivot(index='Нормализация', columns='Модель', values='F1-score')

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
acc_pivot.plot(kind='bar', ax=ax[0], color=['skyblue', 'lightgreen'], rot=0)
ax[0].set_title('Средняя Accuracy по методам нормализации')
ax[0].set_ylabel('Accuracy')
ax[0].legend(title='Модель')
ax[0].set_ylim(0.5, 1.05)

f1_pivot.plot(kind='bar', ax=ax[1], color=['skyblue', 'lightgreen'], rot=0)
ax[1].set_title('Средний F1-score по методам нормализации')
ax[1].set_ylabel('F1-score')
ax[1].legend(title='Модель')
ax[1].set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig('metrics_bar.png')
plt.show()