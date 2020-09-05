import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import sys
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import threading
import time
#import os.path as path
from os import path
from os import remove

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# En caso que se vaya a utilizar la cámara web desomentar
#cap = cv2.VideoCapture(0)

#-----------------------------
#face expression recognizer initialization
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Función utilizada para definir la emoción
def definirEmocion():
    print('Definir emoción')
    ingresar = True
    while(ingresar):
        count = 0 
        vidcap = cv2.VideoCapture("VID_20200314_152552.mp4") 
        success,imgVideo = vidcap.read() 
        success = True 
        while success: 
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # added this line 
            success,imgVideo = vidcap.read() 
            if success:
                print ('Read a new frame: ', success) 		

                # obtiene la anchura y altura
                (h, w) = imgVideo.shape[:2]
                # calcula el centro de la imagen
                center = (w / 2, h / 2)	

                if w > h:
                    # Realice la rotación en sentido antihorario manteniéndose en el centro 90 grados
                    M = cv2.getRotationMatrix2D(center, 90, 1.0)
                    rotated90 = cv2.warpAffine(imgVideo, M, (w, h))
                    # save frame as JPEG file 
                    cv2.imwrite( "/home/developer-carlos/Imágenes/" + "frame%d.jpg" % count, rotated90)
                else:
                    # save frame as JPEG file
                    cv2.imwrite( "/home/developer-carlos/Imágenes/" + "frame%d.jpg" % count, imgVideo) 
                
                # Leemos la imagen que se va a analizar "Se debe definir la imagen a analizar"
                img = cv2.imread("/home/developer-carlos/Imágenes/" + "frame%d.jpg" % count, cv2.IMREAD_COLOR)
                if img.size == 0:
                    sys.exit("Error: la imagen no fue cargada")

                #ret, img = cap.read()	

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
                #faces = face_cascade.detectMultiScale(gray, 1.3, 5)		

                #print(faces) #locations of detected faces

                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
                    
                    print("ingreso")
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                    
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    
                    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                    
                    predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                    
                    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                    max_index = np.argmax(predictions[0])
                            

                    emotion = emotions[max_index]
                    print(emotion)

                print('Escribir')
                f = open('emociones.txt', 'a')
                f.writelines(str(count)+','+emotion+"\n")
                time.sleep(1)
                count = count + 1
            else:        
                success = False        
                ingresar = False

# Función que realiza la gráfica
def animate(i):
    print('Graficar')
    if path.exists('emociones.txt'):
        graph_data = open('emociones.txt', 'r').read()
        lines = graph_data.split('\n')
        xs = []
        xy = []
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                xs.append(x)
                xy.append(y)

        print('XS')
        print(xs)
        print('XY')
        print(xy)
        plt.xlabel('Tiempo')
        plt.ylabel('Emoción')
        plt.plot(xs, xy, color='green')

# Definimos configuración inicial de la gráfica
fig = plt.figure()
#[enojo, disgusto, miedo, triste, neutral, sorpresa, feliz]
emotion = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'surprise', 'happy']
plt.xlabel('Tiempo')
plt.ylabel('Emoción')
xs = [0,0,0,0,0,0,0]
plt.plot(xs, emotion, color='white')

# Se ejecuta un hilo para realizar el proceso de reconocimiento
t = threading.Thread(target=definirEmocion)
t.start()

# Se ejecuta la gráfica animada
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

# Se elimina el archivo 
remove('emociones.txt')