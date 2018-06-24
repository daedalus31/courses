from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15,
                                                    random_state=2)

model = neighbors.KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(model.score(x_test, y_test))

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.scatter(x_test[:, 0], x_test[:, 1], c='m')
plt.show()
