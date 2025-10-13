import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error


def get_demo_classification_data():
    X, Y = make_moons(n_samples=2000, noise=0.3, random_state=0)
    return train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=0)


def get_demo_regression_data():
    np.random.seed(0)
    X = np.random.normal(size=[2000, 5])
    
    NOISE = 0.3 * np.random.normal(size=[2000])
   
    Y = X.mean(axis=1) + (X ** 2).mean(axis=1) + NOISE
   
    return train_test_split(X, Y, test_size=0.4, random_state=0)


# КЛАССИФИКАЦИЯ
X_train, X_test, Y_train, Y_test = get_demo_classification_data()

 
base_model = GradientBoostingClassifier(n_estimators=1, learning_rate=1.0)
base_model.fit(X_train, Y_train)  # Обучаем


gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb_clf.fit(X_train, Y_train)  # Обучаем ансамбль


Y_hat_base = base_model.predict(X_test)  # метки 0/1 от базовой модели
Y_hat = gb_clf.predict(X_test)  # метки от ансамбля

print(f"Классификация: точность одного дерева = {accuracy_score(Y_test, Y_hat_base):.2f}")
print(f"Классификация: точность Gradient Boosting (100 деревьев) = {accuracy_score(Y_test, Y_hat):.2f}")


P_hat = gb_clf.predict_proba(X_test)
# Brier score: средний квадрат ошибки вероятности (меньше — лучше)
print(f"Классификация: мера Бриера = {brier_score_loss(Y_test, P_hat[:, 1]):.2f}")


xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 200),
    np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 200)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, model, title in zip(axes, [base_model, gb_clf],
                            ["Одно дерево", "Gradient Boosting (100 деревьев)"]):
    # Предсказываем класс для каждой точки сетки
    pts = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(pts).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap="bwr", s=10)
    ax.set_title(title)

plt.tight_layout()
plt.show()


# РЕГРЕССИЯ 
X_train_r, X_test_r, Y_train_r, Y_test_r = get_demo_regression_data()

# Модель градиентного бустинга для регрессии
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
gb_reg.fit(X_train_r, Y_train_r)


Y_hat_r = gb_reg.predict(X_test_r)

mse = mean_squared_error(Y_test_r, Y_hat_r)

print(f"Регрессия: MSE = {mse:.3f}")


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], c=Y_test_r, cmap="magma", s=15)
plt.title("Истинные значения (Y)")
plt.colorbar(label="Y")


plt.subplot(1, 3, 2)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], c=Y_hat_r, cmap="coolwarm", s=15)
plt.title("Прогнозы модели (Ŷ)")
plt.colorbar(label="Ŷ")

plt.subplot(1, 3, 3)
errors = Y_test_r - Y_hat_r
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], c=errors, cmap="bwr", s=15)
plt.title("Ошибки (Y - Ŷ)")
plt.colorbar(label="Ошибка")

plt.tight_layout()
plt.show()