import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from kneed import KneeLocator

# Create 2D data (non-linear separable data)
X, y = make_moons(n_samples=300, noise=0.3, random_state=35)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
) 


# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ELBOW METHOD
error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error)

# Find optimal k using KneeLocator
kneedle = KneeLocator(
    x=list(k_range),
    y=error_rates,
    curve='convex',
    direction='decreasing'
)
optimal_k_elbow = kneedle.elbow

# SUBPLOT 1: Elbow Method
ax1.plot(k_range, error_rates, 'bo-', markersize=6, linewidth=2, alpha=0.7)
ax1.axvline(x=optimal_k_elbow, color='red', linestyle='--', linewidth=2, 
           label=f'Optimal k = {optimal_k_elbow}')
ax1.scatter(optimal_k_elbow, error_rates[optimal_k_elbow-1], 
           color='red', s=100, zorder=5)
ax1.set_xlabel('Number of Neighbors (k)')
ax1.set_ylabel('Error Rate')
ax1.set_title('Elbow Method for kNN Classification')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)


## CROSS-VALIDATION
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k_cv = k_range[np.argmax(cv_scores)]
best_score = max(cv_scores)

# SUBPLOT 2: Cross-Validation
ax2.plot(k_range, cv_scores, 'go-', markersize=6, linewidth=2, alpha=0.7)
ax2.axvline(x=best_k_cv, color='red', linestyle='--', linewidth=2,
           label=f'Best k = {best_k_cv}')
ax2.scatter(best_k_cv, best_score, color='red', s=100, zorder=5,
           label=f'Accuracy: {best_score:.3f}')
ax2.set_xlabel('Number of Neighbors: k')
ax2.set_ylabel('Cross-Validated Accuracy')
ax2.set_title('k-NN Cross-Validation Accuracy vs k')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)

plt.tight_layout()
plt.show()