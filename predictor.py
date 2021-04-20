# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 100, 100
path = './modelo/'
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

#cargamos el modelo
cnn = load_model(modelo)
#cargamos los pesos
cnn.load_weights(pesos)

#funcion de prediccion
#entra la imagen
#escrie el resultado
def predict(file):
    #cargamos la imagen
    imagen = load_img(file, target_size=(longitud, altura))
    #transformamos la imagen en array
    imagen = img_to_array(imagen)
    #añadimos una dimension extra en el eje 0
    imagen = np.expand_dims(imagen, axis=0)
    #obtenemos la prediccion del modelo [[1,0,0]] o [[0,1,0]] o [[0,0,1]]
    
    result = cnn.predict(imagen)
    #obtenemos el primer resultado
    result = result[0]
    #obtenemos la posicion en la que esta el 1
    result = np.argmax(result)
    
    if result == 0:
        print("Prediccion: Perro")
    elif result == 1:
        print("Prediccion: Gato")
    elif result == 2:
        print("Prediccion: Gorila")
        
        
print("Para una foto de un gato el modelo predice:")
predict('./data/gato.jpg')
print("Para una foto de un gorila el modelo predice:")
predict('./data/gorila.jpg')
print("Para una foto de un perro el modelo predice:")
predict('./data/perro.jpg')