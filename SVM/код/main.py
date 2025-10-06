# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

wine_data_file = r"C:\Users\user\source\repos\cpp-support-vector-machine\wine_data.csv"
weights_file = r"C:\Users\user\source\repos\cpp-support-vector-machine\weights.csv"
sv_file = r"C:\Users\user\source\repos\cpp-support-vector-machine\support_vectors.csv"

# -------------------------------
df = pd.read_csv(wine_data_file, delimiter=';')
df['type label'] = df['type'].map({1: 'White', -1: 'Red'})

X = df[['param_1', 'param_2']].values
Y = df['type'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_scaled = X
# -------------------------------
x_min, x_max = X_scaled[:,0].min()-0.5, X_scaled[:,0].max()+0.5
y_min, y_max = X_scaled[:,1].min()-0.5, X_scaled[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# -------------------------------
with open(weights_file, 'r') as f:
    weights_line = f.readline().strip()
weights = np.array([float(x) for x in weights_line.split(';') if x != ''])
w0_cpp, w1_cpp, bias_cpp = weights

Z_cpp = np.sign(w0_cpp * grid[:,0] + w1_cpp * grid[:,1] + bias_cpp)
Z_cpp = Z_cpp.reshape(xx.shape)

# -------------------------------
# Чтение опорных векторов из файла (уже стандартизированных)
support_vectors = []
with open(sv_file, 'r') as f:
    for line in f:
        values = line.strip().split(';')
        if len(values) == 2:
            support_vectors.append([float(values[0]), float(values[1])])

support_vectors = np.array(support_vectors)

# -------------------------------
model = LinearSVC(C=0.01, loss='hinge', max_iter=5000, tol=1e-6, random_state=42)
model.fit(X_scaled, Y)
Z_skl = model.predict(grid).reshape(xx.shape)

# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

# C++ LinearSVC
ax1.contourf(xx, yy, Z_cpp, alpha=0.3, levels=[-1,0,1], colors=['red','green'])
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df['type label'], palette={'White':'green', 'Red':'red'}, s=60, alpha=1.0, ax=ax1)

# Добавляем опорные векторы на первый график (уже стандартизированные)
#if len(support_vectors) > 0:
#    ax1.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolor='none', edgecolor='black', linewidth=2, label='Support Vectors')

ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_title('C++ LinearSVC Areas')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Sklearn LinearSVC
ax2.contourf(xx, yy, Z_skl, alpha=0.3, levels=[-1,0,1], colors=['red','green'])
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df['type label'], palette={'White':'green', 'Red':'red'}, s=60, alpha=1.0, ax=ax2)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_title('Sklearn LinearSVC Areas')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()