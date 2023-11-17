from fastapi import FastAPI, Form, UploadFile, File
import base64
from  predecir import predecir_imagen
from PIL import Image
import urllib.request
from detection import PATH_TO_IMAGES, tflite_detect_images, PATH_TO_LABELS, PATH_TO_MODEL, min_conf_threshold, images_to_test
import base64



app = FastAPI()


@app.get("/")
def root():
    return "Hi im fastapi"


@app.post("/imagen")
def imagenRecortada(image: str = Form (...)):
    try:
        
        PATH_TO_IMAGES = base64.b64decode(image)
        cropped_image = tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)
        return str(base64.b64encode(cropped_image))

    except Exception as e:
        print(e)
        return {"message": "Hubo un error al subir la img"}




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