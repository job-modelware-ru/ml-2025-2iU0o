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
# X is a matrix of features. Each row is one point, and the columns are the coordinates (x, y)
# y is a vector of target variables (labels). It contains 0 and 1, indicating which class each point belongs to
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

## Create a DataFrame for plotting
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df['Target'] = y

## Visualize the 2D data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Target", palette="Set1")
plt.title("2D Classification Data (make_moons)")
plt.grid(True)
plt.show()

## Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
) 

k = 15

## Create a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=k)
## Train the model on the training data
knn.fit(X_train, y_train)

## Generate predictions for the test data
y_pred = knn.predict(X_test)
## Compare the predicted labels with the true labels and calculate the accuracy as the proportion of correct predictions.
print(f"Test Accuracy (k={k}): {accuracy_score(y_test, y_pred):.2f}")


# Create a mesh grid for decision boundary visualization
h = 0.02  # step size in the mesh
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Use the trained modelto predict labels for every point in the 2D mesh grid (xx, yy)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predictions (Z) to match the grid's shape for plotting
Z = Z.reshape(xx.shape)

# Create a plot showing the decision boundary by coloring regions according to predicted classes using contourf
plt.figure(figsize=(10, 8))

# Plot decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# Overlay the original data points with different colors representing true classes
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, 
                     edgecolor='k', s=50, alpha=0.8)

plt.title(f"Decision Boundary with k = {k}")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.grid(True, alpha=0.3)
plt.show()
