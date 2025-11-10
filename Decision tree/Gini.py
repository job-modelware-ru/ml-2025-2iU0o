from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score
import pandas as pd
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
models = {
    "ID3 (Entropy)": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "CART (Gini)": DecisionTreeClassifier(criterion='gini', random_state=42),
    "Random Split": DecisionTreeClassifier(criterion='gini', splitter='random', random_state=42),
    "Deep Tree": DecisionTreeClassifier(max_depth=None, random_state=42),
    "Pruned Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42)
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-score": f1_score(y_test, y_pred, average='macro'),
    })

df = pd.DataFrame(results)
print(df)

feature_names = load_iris().feature_names
class_names = load_iris().target_names

for name, model in models.items():
    plt.figure(figsize=(9,6))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names)
    plt.title(name)
    plt.show()