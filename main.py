from typing import Optional
from ml import predict_l1 , predict_l2, predict_l3 
from fastapi import FastAPI
from pydantic import BaseModel


tags_metadata = [
    {
        "name": "Level 1",
    },
    
    {
        "name": "Level 2",
    },
    
    {
        "name": "Level 3",
    },
    
]

app = FastAPI(title="Document Classification API",description="Read about this at https://docs.google.com/document/d/1WYTdadnzqN6W_zw37mKuJRnbyTGoDZvYaUPwOOXUsOU/edit?usp=sharing",openapi_tags=tags_metadata)


@app.get("/")
def read_root():
    return 'Go to /docs'

class query(BaseModel):
    text: str

@app.post("/l1",tags=["Level 1"])
async def create_item(query: query):
    prediction = predict_l1(query.text)
    return {"prediction" : prediction}

@app.post("/l2",tags=["Level 2"])
async def create_item(query: query):
    prediction = predict_l2(query.text)
    return {"prediction" : prediction}

@app.post("/l3",tags=["Level 3"])
async def create_item(query: query):
    prediction = predict_l3(query.text)
    return {"prediction" : prediction}
