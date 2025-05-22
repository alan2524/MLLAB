#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
plt.figure(figsize=(12,10))
df.hist(bins=30,figsize=(12,10),layout=(3,3))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.boxplot(data=df, orient='v')
plt.xticks(rotation=45)
plt.title("Box Plots of All Features")
plt.tight_layout()
plt.show()


# In[15]:


import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load data and create DataFrame
df = pd.DataFrame(
    fetch_california_housing().data,
    columns=fetch_california_housing().feature_names
)

# Correlation heatmap (single line)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm').set_title("Correlation Matrix")

# Pair plot (single line)
sns.pairplot(df)


# In[16]:


import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load data
df = pd.DataFrame(
    fetch_california_housing().data,
    columns=fetch_california_housing().feature_names
)

# --- Histograms (unchanged) ---
df.hist(bins=30, figsize=(12, 10), layout=(3, 3))
plt.suptitle("Histograms of California Housing Features", y=1.02)
plt.tight_layout()
plt.show()

# --- Individual Box Plots ---
plt.figure(figsize=(12, 10))
for i,column in enumerate(df.columns,1):
    plt.subplot(3,3,i)
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot of {column}")
plt.tight_layout()
plt.show()


# In[21]:


import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load, scale, and transform in 3 lines
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(StandardScaler().fit_transform(df)), 
                     columns=['PC1', 'PC2']).assign(Target=load_iris().target)

# Plot in 1 line
sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df).set(
    xlabel='Principal Component 1', ylabel='Principal Component 2', title='PCA of Iris Dataset')


# In[30]:


import pandas as pd
import numpy as np
def find_s_algorithm(data):
    X = data.iloc[:, :-1].values  
    hypothesis = np.array(['?' for _ in range(X.shape[1])])
    for i, row in enumerate(X):
        if y[i] == 'Yes':  
            if '?' in hypothesis:
                hypothesis = row.copy() 
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != row[j]:
                        hypothesis[j] = '?'
    return hypothesis
data = pd.read_csv('training_data.csv')
hypothesis = find_s_algorithm(data)
print("The most specific hypothesis is:", hypothesis)


# In[28]:


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test)*100:.2f}%")
print(f"Prediction: {'Benign' if clf.predict([data.data[0]])[0] else 'Malignant'}")
plot_tree(clf, filled=True, feature_names=list(data.feature_names), class_names=list(data.target_names))
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x_values = np.random.rand(100).reshape(-1, 1)
labels = np.array(["Class1" if x <= 0.5 else "Class2" for x in x_values[:50]])
k_values = [1, 2, 3, 4, 5, 20, 30]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_values[:50], labels)
    predicted_labels = knn.predict(x_values[50:])
    print(f"Results for k={k}:")
    print(predicted_labels)
    print("\n")
plt.scatter(x_values[:50], np.zeros(50), c=["blue" if x <= 0.5 else "red" for x in x_values[:50]], label='Labeled Data')
plt.scatter(x_values[50:], np.zeros(50), c='green', marker='x', label='Unlabeled Data')
plt.xlabel("x values")
plt.title("KNN Classification of Randomly Generated Data")
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg  # Fixed spelling

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.1, size=100)

plt.figure(figsize=(10,6))
for bw in [0.1, 0.5, 1, 5]:  # Changed from [0,0.5,1,5] as bandwidth=0 would cause problems
    kr = KernelReg(y, x, 'c', reg_type='ll', bw=[bw])  # Fixed spelling
    plt.plot(x, kr.fit(x)[0], label=f'bandwidth={bw}')  # Added equals sign for better labeling
    
plt.scatter(x, y, c='k', s=10, label='data')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_plot(features, target, model, title, x_label, y_label):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.scatter(X_test, y_test, c="blue", label="Actual")
    plt.scatter(X_test, y_pred, c="red", label="Predicted")
    plt.xlabel(x_label); plt.ylabel(y_label); plt.title(title)
    plt.legend(); plt.show()
    
    print(f"{title}\nMSE: {mean_squared_error(y_test, y_pred):.2f}\nR²: {r2_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    housing = fetch_california_housing(as_frame=True)
    train_plot(housing.data[["AveRooms"]], housing.target, LinearRegression(),
               "Linear Regression - California Housing",
               "Average Rooms", "Home Value ($100k)")
    
    cars = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                      sep='\s+', names=["mpg","cylinders","displacement","horsepower",
                                      "weight","acceleration","year","origin"],
                      na_values="?").dropna()
    train_plot(cars[["displacement"]], cars["mpg"],
              make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()),
              "Polynomial Regression - Auto MPG",
              "Engine Displacement", "Miles Per Gallon")

# In[ ]:




