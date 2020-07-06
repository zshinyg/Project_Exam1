import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Project_Exam1/Problem2/Customers.csv')

dataset.info()

x = dataset.iloc[:,[3,4]]
y = dataset.iloc[:,-1]

print(x.shape,y.shape)


##ELBOW METHOD
from sklearn.cluster import KMeans
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

##Scaling the data made no difference in Silhouette score.

##KMEANS
from sklearn.cluster import KMeans
nclusters = 5 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

##SILHOUETTE SCORE
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)



