from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

## Create 2D data (non-linear separable data)
X, y = make_moons(n_samples=300, noise=0.3, random_state=35)

## Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
) 

## Different k values to test
k_values = [1, 5, 50, 100]

## Create a mesh grid for decision boundary visualization
h = 0.02  # step size in the mesh
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

## Create a figure with 2x3 subplots - smaller size
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing

## Train models and plot decision boundaries for each k
for i, k in enumerate(k_values):
    ## Create a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    ## Train the model on the training data
    knn.fit(X_train, y_train)
    
    ## Generate predictions for the test data
    y_pred = knn.predict(X_test)
    ## Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    ## Use the trained model to predict labels for every point in the 2D mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    
    ## Reshape the predictions to match the grid's shape for plotting
    Z = Z.reshape(xx.shape)
    
    ## Plot decision boundary
    axes[i].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    ## Overlay the original data points (smaller points)
    scatter = axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, 
                             edgecolor='k', s=15, alpha=0.7)
    
    axes[i].set_title(f"k={k}, Acc:{accuracy:.3f}", fontsize=11, pad=8)
    axes[i].set_xlabel("F1", fontsize=9)
    axes[i].set_ylabel("F2", fontsize=9)
    axes[i].tick_params(axis='both', which='major', labelsize=7)
    axes[i].grid(True, alpha=0.2)

## Add overall title
fig.suptitle("K-NN Decision Boundaries", fontsize=14, y=0.95)

## Adjust layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Less space at the top
fig.tight_layout(pad=3.0)
plt.show()