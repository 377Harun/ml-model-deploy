

from fastapi import FastAPI , requests
import joblib
from data_models import Iris
from data_models import Advertising

estimator_iris_loaded = joblib.load("models/knn_with_iris_dataset.pkl")
encoder_iris_loaded = joblib.load("models/iris_encoder.pkl")
estimator_advertising_loaded = joblib.load("models/randomforest.pkl")

#dir()


app = FastAPI()
 
#iris prediction endpoint 

def make_iris_prediction(model ,encoder ,  request):
    SepalLength = request["Sepallength"]
    SepalWidth = request["SepalWidth"]
    PetalLength = request["Petallength"]
    PetalWidth = request["PetalWidth"]

    flower = [[SepalLength,SepalWidth,PetalLength,PetalWidth]]

    prediction_raw = model.predict(flower)
    prediction_real = encoder.inverse_transform(prediction_raw)
    return prediction_real[0]


@app.post("/prediction/iris")  
async def predict_iris(request: Iris):
    prediction = make_iris_prediction(estimator_iris_loaded , encoder_iris_loaded , request.dict()) 
    return prediction





def make_advertising_prediction(model  ,  request):
    TV = request["TV"]
    Radio = request["Radio"] 
    Newspaper = request["Newspaper"]

    sales = [[TV,Radio,Newspaper]] 
 
    prediction = model.predict(sales)
    return prediction[0] 
 

@app.post("/prediction/advertising")     
async def predict_advertising(request: Advertising): 
    prediction = make_advertising_prediction(estimator_advertising_loaded  , request.dict())
    #burasi yalnizca deger dondurur index 0 almana gerek yok zaten make_advertising_prediction fonksiyonunda
    # 0 indexi donduk
    return {"sonuc": round(prediction,2) ,"yazar":"harun bakirci"} 




