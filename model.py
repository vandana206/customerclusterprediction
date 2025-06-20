# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

# Loading the Dataset into DataFrame
df = pd.read_csv("Mall_Customers1.csv")

# Selecting relevant features
x = df[["Annual Income (k$)", "Spending Score (1-100)"]]  # FIXED COLUMN NAME

# Using Elbow method to find optimal number of clusters
wcss_list = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=1)
    model.fit(x)
    wcss_list.append(model.inertia_)

# Visualize the Elbow result
#plt.plot(range(1, 11), wcss_list, marker='o')
#plt.title("Elbow Method Graph")
#plt.xlabel("Number of Clusters")
#plt.ylabel("WCSS")
#plt.show()
"""
# Training the model
model = KMeans(n_clusters=5,init="k-means++",random_state=1)
y_predict = model.fit_predict(x)
print(y_predict)

#converting the dataframe x into a numpy array
x_array = x.values
"""
# plotting the graph of clusters
"""
plt.scatter(x_array[y_predict == 0, 0],x_array[y_predict ==0,1],s = 100,color = "Green")
plt.scatter(x_array[y_predict == 1, 0],x_array[y_predict ==1,1],s = 100,color = "Red")
plt.scatter(x_array[y_predict == 2, 0],x_array[y_predict ==2,1],s = 100,color = "Yellow")
plt.scatter(x_array[y_predict == 3, 0],x_array[y_predict ==3,1],s = 100,color = "Blue")
plt.scatter(x_array[y_predict == 4, 0],x_array[y_predict ==4,1],s = 100,color = "pink")


plt.title("Customer Segmentation Graph")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

"""
joblib.dump(model,"Model.pkl")
print("Model has been save")