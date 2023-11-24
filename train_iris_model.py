
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


iris = datasets.load_iris()
print(iris["data"])

print(iris)
irisX = pd.DataFrame( iris["data"] ,columns=["sepal.length","sepal.width","petal.length","petal.width"])
irisY = pd.DataFrame(iris["target"], columns=["species"])

iris = pd.concat([irisX  , irisY],axis=1)

iris.shape
iris.describe().T


from sklearn.preprocessing import LabelEncoder
#bu encoder amacı string degerleri int'a cevirmek
a = ["harun","ahmet","samet","harun","kasim"]
encoder = LabelEncoder()
y = encoder.fit_transform(a)
encoder.inverse_transform(y)


import joblib

joblib.dump(encoder , "models/iris_encoder.pkl")


X_train , X_test , y_train, y_test = train_test_split(irisX,irisY, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

#encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test , y_pred=y_pred)

joblib.dump(model , "models/knn_with_iris_dataset.pkl")

'''
knn_model = joblib.load("models/knn_with_iris_dataset.pkl")
result = knn_model.predict([[1,2,2,1], [2.1,3.2,4,4]])
classifierEncoder = joblib.load("models/iris_encoder.pkl")
classifierEncoder.inverse_transform(result)
'''

