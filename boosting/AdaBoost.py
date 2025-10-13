import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Генерация 2-х класса точек с шумом
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# Делим выборку: 60% для обучения, 40% для проверки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Строим один "пень" (дерево глубины 1 — очень простое)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)  # учим на тренировочных данных
y_pred_stump = stump.predict(X_test)  # предсказываем
print(f"Классификация: точность одного пня = {accuracy_score(y_test, y_pred_stump):.2f}")

# Строим AdaBoost из 50 "пней"
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train, y_train)  # учим ансамбль
y_pred_ada = ada.predict(X_test)  # предсказываем
print(f"Классификация: точность AdaBoost (50 пней) = {accuracy_score(y_test, y_pred_ada):.2f}")


def plot_decision_boundary(model, X, y, title):
    # Определяем границы области для рисования
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Создаём сетку точек (200×200) покрывающую область
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Для каждой точки на сетке предсказываем класс
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Рисуем фон (классы разными цветами)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Рисуем реальные точки
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')

    plt.title(title)
    plt.show()


# Рисуем для одного пня
plot_decision_boundary(stump, X_train, y_train, "Один пень (смещение большое)")
# Рисуем для AdaBoost
plot_decision_boundary(ada, X_train, y_train, "AdaBoost (смещение уменьшилось)")
# Генерация данных: y = x^2 + шум
rng = np.random.RandomState(42)
X = np.linspace(-2, 2, 200).reshape(-1, 1)  # 200 точек от -2 до 2
y = X.ravel() ** 2 + rng.normal(0, 0.3, size=200)  # добавляем случайный шум

# Делим: первые 150 для train, последние 50 для test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Один "пень" для регрессии
stump_reg = DecisionTreeRegressor(max_depth=1, random_state=42)
stump_reg.fit(X_train, y_train)
y_pred_stump = stump_reg.predict(X_test)
print("Регрессия: MSE одного пня =", mean_squared_error(y_test, y_pred_stump))
# MSE = среднее (ошибка^2) — чем меньше, тем лучше
# AdaBoost для регрессии из 100 деревьев глубины 3
ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    random_state=42
)
ada_reg.fit(X_train, y_train)
y_pred_ada = ada_reg.predict(X_test)
print("Регрессия: MSE AdaBoost =", mean_squared_error(y_test, y_pred_ada))
# Рисуем точки train (синие) и test (красные)
plt.scatter(X_train, y_train, color="blue", s=10, label="train data")
plt.scatter(X_test, y_test, color="red", s=10, label="test data")

# Создаём много точек для гладкой линии
X_plot = np.linspace(-2, 2, 400).reshape(-1, 1)

# Предсказания одного пня (зелёная линия)
plt.plot(X_plot, stump_reg.predict(X_plot), color="green", label="Один пень")

# Предсказания AdaBoost (чёрная линия)
plt.plot(X_plot, ada_reg.predict(X_plot), color="black", label="AdaBoost (100 пней)")

plt.legend()
plt.title("Regressor: уменьшение смещения")
plt.show()
