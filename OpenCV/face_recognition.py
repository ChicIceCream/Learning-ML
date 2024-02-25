import numpy as np
import cv2 as cv

def rescaleFrame(frame, scale=1.5):
    width = int(frame[1] * scale)
    height = int(frame[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']

features = np.load('OpenCV/features.npy', allow_pickle=True)
labels = np.load('OpenCV/labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('OpenCV/face_trained_model.yml')

img = cv.imread(
    r'OpenCV\Faces\val\ben_afflek\httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTENDgMDUODczNDcNTcjpg.jpg'
)
img = rescaleFrame(img, 1.5)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    
    cv.putText(img, str(people[label]), (10,10), cv.FONT_HERSHEY_COMPLEX, 1.0,(0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Detected Face", img)

cv.waitKey(0)
