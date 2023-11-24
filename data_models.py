


from pydantic import BaseModel


class Iris(BaseModel):
    Sepallength : float
    SepalWidth : float
    Petallength : float
    PetalWidth : float

    class Config:
        schema_extra = { 
            "example":{
                "Sepallength":2.1,
                "SepalWidth": 3.3, 
                "Petallength":1.1,
                "PetalWidth":4.2
            }
        }

class Advertising(BaseModel): 
    TV : float
    Radio : float
    Newspaper : float

    class Config:
        schema_extra = { 
            "example":{ 
                "TV":2.1,
                "Radio": 3.3, 
                "Newspaper":1.1,
            }
        }







