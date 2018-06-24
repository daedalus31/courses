from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris= datasets.load_iris()
x = iris.data[:, 1:3]

model = KMeans(n_clusters=3, random_state=0)
model.fit(x)

centroids = model.cluster_centers_

plt.scatter(centroids[:,0], centroids[:, 1], marker='^', s=170, c='r', zorder=10)
plt.scatter(x[:, 0], x[:, 1], c=model.labels_)
plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()
