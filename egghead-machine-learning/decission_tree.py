from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz

breast_cancer = datasets.load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=2)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(x_test)

print(model.score(x_test, y_test))

print(metrics.classification_report(y_test, y_predict))
print(metrics.confusion_matrix(y_test, y_predict))

graph_data = tree.export_graphviz(model, out_file=None,
                                  feature_names=breast_cancer.feature_names,
                                  filled=True)
graph = graphviz.Source(graph_data)
graph.render("breast_cancer", view=True)
