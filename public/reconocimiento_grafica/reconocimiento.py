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
import os
#import os.path as path
from os import path
from os import remove
import pandas as pd

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

ssd_detector = cv2.dnn.readNetFromCaffe("SSD-deploy.prototxt.txt", "SSD-res10_300x300_ssd_iter_140000.caffemodel")
ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

# Función utilizada para definir la emoción
def definirEmocion(video):
    print('Definir emoción')
    ingresar = True
    while(ingresar):
        count = 0 
        num = 0
        vidcap = cv2.VideoCapture(video) 
        success,img = vidcap.read() 
        success = True 
        definiteEmotion = False
        while success: 
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # added this line 
            success,img = vidcap.read() 
            if success:
                print ('Read a new frame: ', success) 		
                '''
                # obtiene la anchura y altura
                (h, w) = imgVideo.shape[:2]
                # calcula el centro de la imagen
                center = (w / 2, h / 2)	

                #if w > h:
                    #print("rotar")
                    # Realice la rotación en sentido antihorario manteniéndose en el centro 90 grados
                    #M = cv2.getRotationMatrix2D(center, 90, 1.0)
                    #rotated90 = cv2.warpAffine(imgVideo, M, (w, h))
                    # save frame as JPEG file 
                    #cv2.imwrite( "/home/developer-carlos/Imágenes/" + "frame%d.jpg" % num, rotated90)
                #else:
                    # save frame as JPEG file
                
                cv2.imwrite( "/home/developer-carlos/Imágenes/" + "frame%d.jpg" % num, imgVideo) 
                
                # Leemos la imagen que se va a analizar "Se debe definir la imagen a analizar"
                img = cv2.imread("/home/developer-carlos/Imágenes/" + "frame%d.jpg" % num, cv2.IMREAD_COLOR)
                if img.size == 0:
                    sys.exit("Error: la imagen no fue cargada")

                #ret, img = cap.read()	

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
                #faces = face_cascade.detectMultiScale(gray, 1.3, 5)		

                print(faces) #locations of detected faces
                '''
                
                original_size = img.shape
                target_size = (300, 300)
                base_img = img.copy() #high resolution image
                img = cv2.resize(img, target_size)
                
                aspect_ratio_x = (original_size[1] / target_size[1])
                aspect_ratio_y = (original_size[0] / target_size[0])
                
                imageBlob = cv2.dnn.blobFromImage(image = img)
                
                ssd_detector.setInput(imageBlob)
                detections = ssd_detector.forward()
                detections_df = pd.DataFrame(detections[0][0], columns = ssd_labels)
                detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
                detections_df = detections_df[detections_df['confidence'] >= 0.90]
                
                detections_df['left'] = (detections_df['left'] * 300).astype(int)
                detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
                detections_df['right'] = (detections_df['right'] * 300).astype(int)
                detections_df['top'] = (detections_df['top'] * 300).astype(int)

                for i, instance in detections_df.iterrows():
                    confidence_score = str(round(100*instance["confidence"], 2))+" %"
                    left = instance["left"] #izquierda
                    right = instance["right"] # derecha
                    bottom = instance["bottom"] # fondo
                    top = instance["top"] #cima
                    
                    detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

                    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                        
                        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                        
                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis = 0)
                        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                        
                        custom = model.predict(img_pixels)
                        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                        
                        print ("predicciones ", predictions)
                        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                        max_index = np.argmax(predictions[0])
                        v_max_index ="{0:.3f}".format( predictions[0,max_index])
                        v_enojo = "{0:.3f}".format(predictions[0,0])
                        v_disgusto = "{0:.3f}".format(predictions[0,1])
                        v_miedo = "{0:.3f}".format(predictions[0,2])
                        v_feliz = "{0:.3f}".format(predictions[0,3])
                        v_tristeza = "{0:.3f}".format(predictions[0,4])
                        v_sorpresa = "{0:.3f}".format(predictions[0,5])
                        v_neutral = "{0:.3f}".format(predictions[0,6])

                        definiteEmotion = True
                        #v_final = str(v_max_index+" sisas ")

                '''
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
                    v_max_index ="{0:.3f}".format( predictions[0,max_index])
                    v_enojo = "{0:.3f}".format(predictions[0,0])
                    v_disgusto = "{0:.3f}".format(predictions[0,1])
                    v_miedo = "{0:.3f}".format(predictions[0,2])
                    v_feliz = "{0:.3f}".format(predictions[0,3])
                    v_tristeza = "{0:.3f}".format(predictions[0,4])
                    v_sorpresa = "{0:.3f}".format(predictions[0,5])
                    v_neutral = "{0:.3f}".format(predictions[0,6])        

                    definiteEmotion = True

                    #emotion = emotions[max_index]
                    #print(emotion)
                '''

                f = open('emociones.txt', 'a')
                if definiteEmotion:
                    #print('Escribir')
                    #print(str(num)+','+v_enojo+','+v_disgusto+','+v_miedo+','+v_feliz+','+v_tristeza+','+v_sorpresa+','+v_neutral+"\n")                    
                    f.writelines(str(num)+','+v_enojo+','+v_disgusto+','+v_miedo+','+v_feliz+','+v_tristeza+','+v_sorpresa+','+v_neutral+"\n")
                else:
                    f.writelines(str(num)+','+'0.0'+','+'0.0'+','+'0.0'+','+'0.0'+','+'0.0'+','+'0.0'+','+'0.0'+"\n")

                time.sleep(1)
                count = count + 1
                num = num + 1
                definiteEmotion = False
            else:        
                success = False        
                ingresar = False

#print("Ingreso")
#print("Argumento principal:",sys.argv[1])
if os.path.isfile('emociones.txt'):
    remove('emociones.txt')
definirEmocion(sys.argv[1])
sys.exit()
