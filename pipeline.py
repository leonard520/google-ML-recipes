from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print accuracy_score(y_test, prediction)

nbrs = KNeighborsClassifier()
nbrs = nbrs.fit(x_train, y_train)
prediction = nbrs.predict(x_test)
print accuracy_score(y_test, prediction)
