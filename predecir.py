import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests



altura, longitud = 96, 96
model = tf.keras.models.load_model('model.h5')



def predecir_imagen(ruta_imagen):
   img = Image.open(requests.get(ruta_imagen, stream = True).raw).resize((altura, longitud))
   image_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
   image_array = np.expand_dims(image_array, axis=0)

   prediction = model.predict(image_array)[0][0]
   #probabilidades = prediccion[0] * 100
   label = "melanoma" if prediction >= 0.5 else "no melanoma"

   print("The probability of melanoma is:", prediction)
   print("Prediction:", label)
   return prediction