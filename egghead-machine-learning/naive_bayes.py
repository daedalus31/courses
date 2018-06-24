from sklearn import datasets, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

newsgropus_train = datasets.fetch_20newsgroups(subset='train')
newsgropus_test = datasets.fetch_20newsgroups(subset='test')

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(newsgropus_train.data)
x_test = vectorizer.transform(newsgropus_test.data)
y_train = newsgropus_train.target
y_test = newsgropus_test.target

model = MultinomialNB()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(model.score(x_test, y_test))
print(metrics.classification_report(y_test, prediction))

labels = list(newsgropus_train.target_names)
cm = ConfusionMatrix(y_test, prediction, labels)
cm.plot()
plt.show()
