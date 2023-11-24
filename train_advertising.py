

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("advertising.csv")

X = df.drop("Sales",axis=1)

y = df.Sales


train_x, test_x , train_y,test_y = train_test_split(X , y , test_size=0.3)


'''

df.head()


df.isnull().sum()

def minMaxsum(df):
    return df.min() + df.max()

k = pd.cut(df.TV , [0,100,150,200, 300])

#df.describe()


df.shape
df.ndim
df.info()


print(train_x.shape)
print(test_x.shape)



'''

model = RandomForestRegressor(n_estimators=200)
model.fit(train_x, train_y)

sonuc = model.predict(test_x)

dogruluk = r2_score(test_y , sonuc)



import joblib
joblib.dump(model , "models/randomforest.pkl")



#model okuma 
a = df.sample(5)
readedModel = joblib.load("models/randomforest.pkl")
readedPredict = readedModel.predict(a[["TV","Radio","Newspaper"]])





