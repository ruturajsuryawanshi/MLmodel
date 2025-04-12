# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'C:\Desktop\MLmodel\Iris.csv')

# Drop the 'Id' column
df = df.drop(columns=['Id'])

# Display basic information
print(df.describe())
print(df.info())

# Display number of samples per class
print(df['Species'].value_counts())

# Check for null values
print(df.isnull().sum())

# Histograms for feature distribution
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()
plt.show()

# Scatter plots for class-wise separation
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

# Sepal Length vs Sepal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

# Petal Length vs Petal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()

# Sepal Length vs Petal Length
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.show()

# Sepal Width vs Petal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
plt.show()

# Model training and testing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Define input and output data
X = df.drop(columns=['Species'])
Y = df['Species']

# Spliting dataaset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=51)

# Decision Tree 
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
print("Decision Tree Accuracy: ", dt_model.score(x_test, y_test) * 100)

# K-nn
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
print("k-NN Accuracy: ", knn_model.score(x_test, y_test) * 100)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_model, X, Y, cv=5)
print("Cross-Validation Accuracy:", scores.mean() * 100)

# Save the Decision Tree model
filename = 'iris_model.pkl'
pickle.dump(dt_model, open(filename, 'wb'))
