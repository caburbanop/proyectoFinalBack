import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import dlib
from mtcnn import MTCNN
#-------------------------------------- emotions
from keras.preprocessing import image

#--------------------------------------------

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
	eye_detector_path = path+"/data/haarcascade_eye.xml"
	
	if os.path.isfile(face_detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
	
	return path+"/data/"

#--------------------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('Enojo', 'disgusto', 'Miedo', 'Feliz', 'Tristeza', 'sorpresa', 'neutral')
    y_pos = np.arange(len(objects))# crea matriz que comienza en cero, incrementa en 1 y termina en el tamaño de objets
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
#------------------------------

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier( '/Users/MrRabbit/Downloads/Emotion-detection-master/Tensorflow/haarcascade_frontalface_default.xml' )
#cap = cv2.VideoCapture(0) #captura la imagen y se guarda en variable cap
cap = cv2.VideoCapture('migue_prueba1.mp4') #captura de video.
salida = cv2.VideoWriter('videoMigue.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(1920,1080))
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------
emotions = ('Enojo', 'disgusto', 'Miedo', 'Feliz', 'Tristeza', 'sorpresa', 'neutral')

#opencv ssd
#model structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
#pre-trained weights: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
ssd_detector = cv2.dnn.readNetFromCaffe("/Users/MrRabbit/Documents/OpenCv/proyectos_Opencv/RecFacial-Python/SSD-deploy.prototxt.txt", "/Users/MrRabbit/Documents/OpenCv/proyectos_Opencv/RecFacial-Python/SSD-res10_300x300_ssd_iter_140000.caffemodel")
ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

#opencv haar cascade
opencv_path = get_opencv_path()
haar_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
haar_detector = cv2.CascadeClassifier(haar_detector_path)

#dlib
dlib_detector = dlib.get_frontal_face_detector()

#dlib cnn
#pre-trained model: http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

#mtcnn
mtcnn_detector = MTCNN()

#--------------------------------------------

detector_models = ['ssd', 'haar', 'dlib', 'mtcnn', 'dlib_cnn']
detector_model = detector_models[0]
print("Detector: ", detector_model)

#--------------------------------------------

#cap = cv2.VideoCapture('migue-.mp4')
cap = cv2.VideoCapture(0)
#salida = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(1920,1080))


#--------------------------------------------

quit = False
tic = time.time()
frame = 0
while(True):
	
	if frame % 100 == 0:
		toc = time.time()
		print(frame,", ",toc-tic)
		tic = time.time()
	
	ret, img = cap.read()
	
	try:
		original_size = img.shape
		
		cv2.putText(img, detector_model, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
	
		

















		if detector_model == 'ssd':
			
			#img = cv2.imread('img1.jpg')
			
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
					v_final = str(v_max_index+" sisas ")

								
								
								
								# 

								
								#img = base_img.copy()
					emotion = emotions[max_index]

					cv2.putText(base_img, emotion+" "+v_final, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
					cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image

					#cv2.putText(img, emotion+" "+str(v_max_index)+"%", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
				
	















		elif detector_model == 'haar':
			
			#img = cv2.imread('img1.jpg')
			base_img = img.copy()## nuevo----------------
			
			#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
			#nuevo----------------
			
			faces = haar_detector.detectMultiScale(gray, 1.3, 5)
			
			for (x,y,w,h) in faces:

				if w > 0:
					

					#confidence_score = str("99.9"+" %")

					cv2.rectangle(base_img, (x,y), (x+w,y+h),(255,255,255), 1) #highlight detected face
					#cv2.putText(base_img, confidence_score, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		elif detector_model == 'dlib':
			
			#img = dlib.load_rgb_image("img1.jpg")
			base_img = img.copy()
			detections = dlib_detector(img, 1)
			#print("detected faces: ",len(detections))
			
			for idx, d in enumerate(detections):

				#confidence_score = str(round(100*d.confidence, 2))+'%'
				left = d.left(); right = d.right()
				top = d.top(); bottom = d.bottom()
				
				cv2.rectangle(base_img, (left, top), (right, bottom),(255,255,255), 1) #highlight detected face
				#cv2.putText(base_img, confidence_score, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		elif detector_model == 'dlib_cnn':
			base_img = img.copy()
			detections = cnn_face_detector(img, 1)
			
			for idx, d in enumerate(detections):
				
				confidence_score = str(round(100*d.confidence, 2))+'%'
				left = d.rect.left()
				right = d.rect.right()
				top = d.rect.top()
				bottom = d.rect.bottom()
				
				cv2.putText(img, confidence_score, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
					
				cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 1) 
		
		elif detector_model == 'mtcnn':
			base_img = img.copy()
			#img = dlib.load_rgb_image("img1.jpg")
						
			detections = mtcnn_detector.detect_faces(img)
			
			for detection in detections:
				confidence_score = str(round(100*detection["confidence"], 2))+"%"
				x, y, w, h = detection["box"]
				
				cv2.putText(base_img, confidence_score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
				
				cv2.rectangle(base_img, (x, y), (x+w, y+h),(255,255,255), 1) #highlight detected face
			
		#----------------------------
	
		#cv2.imshow('img',img)
		cv2.imshow('img b',base_img)

		#salida.write(base_img)
		cv2.imwrite( "%s/%s.jpg" % (detector_model, str(frame)), img );
		frame = frame + 1
	except Exception as e:
		quit = True
		print(str(e))
	
	if quit == True or (cv2.waitKey(1) & 0xFF == ord('q')): #press q to quit
		toc = time.time()
		
		print(frame," frames process in ",toc-tic," seconds")
		break

#kill open cv things
cap.release()
#salida.release()
cv2.destroyAllWindows()