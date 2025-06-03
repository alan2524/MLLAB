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
for i,column in enumerate(df.columns,1):
    plt.subplot(3,3,i)
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot of {column}")
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
df = pd.DataFrame(
    fetch_california_housing().data,
    columns=fetch_california_housing().feature_names
)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm').set_title("Correlation Matrix")
sns.pairplot(df)




import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(StandardScaler().fit_transform(df)), 
                     columns=['PC1', 'PC2']).assign(Target=load_iris().target)

sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df).set(
    xlabel='Principal Component 1', ylabel='Principal Component 2', title='PCA of Iris Dataset')



import pandas as pd
import numpy as np

def find_s(data):
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    hypothesis = ['?'] * X.shape[1]

    for i in range(len(X)):
        if y[i] == 'Yes':
            instance = X.iloc[i].values
            if '?' in hypothesis:
                hypothesis = instance
            else:
                hypothesis = [h if h == x else '?' for h, x in zip(hypothesis, instance)]

    return hypothesis

data = pd.read_csv('training_data.csv')
print("The most specific hypothesis is:", find_s(data))




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
from statsmodels.nonparametric.kernel_regression import KernelReg 

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.1, size=100)

plt.figure(figsize=(10,6))
for bw in [0.1, 0.5, 1, 5]:  
    kr = KernelReg(y, x, 'c', reg_type='ll', bw=[bw])  
    plt.plot(x, kr.fit(x)[0], label=f'bandwidth={bw}') 
    
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


import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
print(f"Accuracy: {clf.score(Xte, yte)*100}%")

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True)
plt.show()

sample = np.array([[14, 20.5, 90, 600, 0.1, 0.2, 0.3, 0.15, 0.25, 0.05,
                    0.5, 1, 3, 40, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003,
                    16, 25, 100, 800, 0.15, 0.3, 0.4, 0.2, 0.3, 0.07]])
print("Prediction:", data.target_names[clf.predict(sample)[0]])


import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = fetch_olivetti_faces(random_state=42)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred) * 100}%")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
X=StandardScaler().fit_transform(load_breast_cancer().data)
kmeans=KMeans(n_clusters=2).fit(X)
x_pca=PCA(n_components=2).fit_transform(X)
centroids=PCA(n_components=2).fit(X).transform(kmeans.cluster_centers_)
plt.scatter(*x_pca.T,c=kmeans.labels_,cmap="viridis")
plt.scatter(*centroids.T, c="red",s=200,marker="x")
plt.show()





