from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("iris9.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
classifier = KNeighborsClassifier(n_neighbors=8, p=3, metric='euclidean')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix is as follows\n', cm)
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))
print("Correct prediction", accuracy_score(y_test, y_pred))
print("Wrong prediction", (1 - accuracy_score(y_test, y_pred)))
