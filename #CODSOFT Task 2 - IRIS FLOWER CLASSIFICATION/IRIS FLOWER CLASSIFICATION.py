import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import missingno as msno

df=pd.read_csv(r'E:\lab\project\Data_Science\Codsoft_task 3\IRIS.csv')
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

msno.matrix(df)
plt.title("Missing Data Matrix")
plt.show()

print(df.duplicated().sum())

df = df.drop_duplicates()
print(df.duplicated().sum())
print(df.describe())
df2 = df.drop('species',axis=1)
df3 = df.copy()

corr_matrix = df2.corr()
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
colors = ['#ffd59e', '#f44336', '#ffaa3d']  
colormap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=colormap,
            cbar_kws={"shrink": .5}, square=True,
            linewidths=1, linecolor='white')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Plot")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['sepal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['sepal_width'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['petal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['petal_width'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Petal Width")
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['species'], bins=20, color='skyblue', edgecolor='black')
plt.title("Bar diagram of Species")
plt.xlabel("Species")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.boxplot(x="species", y="sepal_length", data=df, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length by Species')
sns.boxplot(x="species", y="sepal_width", data=df, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width by Species')
sns.boxplot(x="species", y="petal_length", data=df, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length by Species')
sns.boxplot(x="species", y="petal_width", data=df, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width by Species')
plt.tight_layout()
plt.show()

X = df.drop('species',axis=1)
y = df['species']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=25)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy_score(Y_test,y_pred)

scores = []
for i in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(Y_test,y_pred))
print(scores)

plt.plot(range(1,16),scores)
plt.xlabel("values of K")
plt.ylabel("Accuracy Scores")
plt.show()