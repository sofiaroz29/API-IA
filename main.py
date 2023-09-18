from fastapi import FastAPI, Form, UploadFile, File
import base64
from  predecir import predecir_imagen
from PIL import Image
import urllib.request



app = FastAPI()


@app.get("/")
def root():
    return "Hi im fastapi"


@app.post("/upload")
def result(image_url:  str = Form(...)):
    
    try:
        resultado = predecir_imagen(image_url)
        porcentaje = round(resultado * 100, 2)
        print(resultado)
        print(porcentaje)
         
    except Exception as e:
        print(e)
        return {"message": "Hubo un error al subir la img"}

    return str(porcentaje)