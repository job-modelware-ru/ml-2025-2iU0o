import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from packaging import version
import sklearn

RANDOM_STATE = 42

def show_bar_with_errorbars(labels, means, stds, title, y_min=None, y_max=None):
    import numpy as np
    x = np.arange(len(means))
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels, rotation=20, ha="right")

    lo = np.min(means - stds)
    hi = np.max(means + stds)
    if y_min is None: y_min = lo
    if y_max is None: y_max = hi
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    plt.ylim(y_min - pad, y_max + pad)
    plt.tight_layout()
    plt.show()

def classification_demo():
    print("\n==============================")
    print("КЛАССИФИКАЦИЯ — IRIS")
    print("==============================")

    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    train_scores, test_scores = [], []
    depths = range(1, 13)
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_scores.append(m.score(X_train, y_train))  # всегда растёт
        test_scores.append(m.score(X_test, y_test))  # после порога падает

    plt.figure(figsize=(6, 4))
    plt.plot(depths, train_scores, marker='o', label='Train accuracy')
    plt.plot(depths, test_scores, marker='s', label='Test accuracy')
    plt.xlabel('max_depth');
    plt.ylabel('Accuracy')
    plt.title('Overfitting: train vs test (Decision Tree, Iris)')
    plt.legend();
    plt.grid(True, alpha=0.4);
    plt.tight_layout();
    plt.show()

    # -- Decision Tree
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print("\n-- Decision Tree --")
    print("Accuracy:", accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt, target_names=class_names))

    # -- Random Forest: влияние bagging и random feature selection
    configs = [
        ("bootstrap=True, max_features='sqrt'", dict(bootstrap=True,  max_features="sqrt")),
        ("bootstrap=True, max_features=None",   dict(bootstrap=True,  max_features=None)),
        ("bootstrap=False, max_features='sqrt'",dict(bootstrap=False, max_features="sqrt")),
        ("bootstrap=False, max_features=None",  dict(bootstrap=False, max_features=None)),
    ]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)

    means, stds, labels = [], [], []
    for label, params in configs:
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, **params)
        scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        means.append(scores.mean()); stds.append(scores.std()); labels.append(label)
        print(f"RF [{label}] — CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    show_bar_with_errorbars(labels, means, stds,
                            "Классификация: влияние bagging и random feature selection (CV accuracy)",
                            y_min=0.9, y_max=1.0)


    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", bootstrap=True,
                                random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("\n-- Random Forest (bootstrap=True, max_features='sqrt') --")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

    # Матрицы ошибок
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), display_labels=class_names)\
        .plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Decision Tree — матрица ошибок")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=class_names)\
        .plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Random Forest — матрица ошибок")
    plt.tight_layout()
    plt.show()

    # Важность признаков
    idx = np.argsort(rf.feature_importances_)[::-1]
    plt.figure(figsize=(7, 4))
    plt.title("Iris: важность признаков (Random Forest)")
    plt.bar(range(len(idx)), rf.feature_importances_[idx])
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45)
    plt.tight_layout(); plt.show()

    # Визуализация одного дерева из леса
    rf_small = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=RANDOM_STATE)
    rf_small.fit(X_train, y_train)
    plt.figure(figsize=(12, 7))
    plot_tree(rf_small.estimators_[0], feature_names=feature_names, class_names=class_names,
              filled=True, impurity=True, rounded=True)
    plt.title("Одно из деревьев из Random Forest (max_depth=3)")
    plt.show()

    # Влияние глубины
    depths = list(range(1, 13))
    rf_scores, dt_scores = [], []
    for d in depths:
        m_rf = RandomForestClassifier(n_estimators=200, max_depth=d, random_state=RANDOM_STATE)
        m_rf.fit(X_train, y_train); rf_scores.append(m_rf.score(X_test, y_test))
        m_dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        m_dt.fit(X_train, y_train); dt_scores.append(m_dt.score(X_test, y_test))

    plt.figure(figsize=(6, 4))
    plt.plot(depths, dt_scores, marker='o', label="Decision Tree")
    plt.plot(depths, rf_scores, marker='s', label="Random Forest")
    plt.xlabel("max_depth"); plt.ylabel("Accuracy (test)")
    plt.title("Iris: влияние глубины на точность")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()



def regression_demo():
    print("\n==============================")
    print("РЕГРЕССИЯ — DIABETES")
    print("==============================")

    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )


    dtr = DecisionTreeRegressor(random_state=RANDOM_STATE)
    dtr.fit(X_train, y_train)
    pred_dt = dtr.predict(X_test)

    def print_regression_report(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"=== {name} ===")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R^2 : {r2:.3f}")

    print_regression_report("Decision Tree (reg)", y_test, pred_dt)

    # Random Forest Regressor — влияние bagging / max_features
    configs = [
        ("bootstrap=True, max_features='sqrt'", dict(bootstrap=True,  max_features="sqrt")),
        ("bootstrap=True, max_features=None",   dict(bootstrap=True,  max_features=None)),
        ("bootstrap=False, max_features='sqrt'",dict(bootstrap=False, max_features="sqrt")),
        ("bootstrap=False, max_features=None",  dict(bootstrap=False, max_features=None)),
    ]
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)

    means, stds, labels = [], [], []
    for label, params in configs:
        rfr = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, **params)
        scores = cross_val_score(rfr, X, y, cv=cv, scoring="r2")
        means.append(scores.mean()); stds.append(scores.std()); labels.append(label)
        print(f"RF [{label}] — CV R^2: {scores.mean():.3f} (+/- {scores.std():.3f})")

    show_bar_with_errorbars(labels, means, stds,
                            "Регрессия: влияние bagging и random feature selection (CV R²)")

    # Выбранный RF
    rfr = RandomForestRegressor(n_estimators=400, max_features="sqrt", bootstrap=True,
                                random_state=RANDOM_STATE)
    rfr.fit(X_train, y_train)
    pred_rf = rfr.predict(X_test)
    print_regression_report("Random Forest (reg)", y_test, pred_rf)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, pred_rf, alpha=0.7)
    plt.xlabel("Фактические значения"); plt.ylabel("Предсказанные (RF)")
    plt.title("Diabetes: фактические vs предсказанные")
    lims = [min(y_test.min(), pred_rf.min()), max(y_test.max(), pred_rf.max())]
    plt.plot(lims, lims)
    plt.tight_layout(); plt.show()

    # Важность признаков
    idx = np.argsort(rfr.feature_importances_)[::-1]
    plt.figure(figsize=(7, 4))
    plt.title("Diabetes: важность признаков (Random Forest Regressor)")
    plt.bar(range(len(idx)), rfr.feature_importances_[idx])
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45)
    plt.tight_layout(); plt.show()

    # Влияние глубины
    depths = list(range(1, 21))
    rf_scores_r2, dt_scores_r2 = [], []
    for d in depths:
        m_rf = RandomForestRegressor(n_estimators=400, max_depth=d, random_state=RANDOM_STATE)
        m_rf.fit(X_train, y_train); rf_scores_r2.append(m_rf.score(X_test, y_test))
        m_dt = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        m_dt.fit(X_train, y_train); dt_scores_r2.append(m_dt.score(X_test, y_test))

    plt.figure(figsize=(6, 4))
    plt.plot(depths, dt_scores_r2, marker='o', label="Decision Tree (R^2)")
    plt.plot(depths, rf_scores_r2, marker='s', label="Random Forest (R^2)")
    plt.xlabel("max_depth"); plt.ylabel("R^2 (test)")
    plt.title("Diabetes: влияние глубины на качество")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    classification_demo()
    regression_demo()
