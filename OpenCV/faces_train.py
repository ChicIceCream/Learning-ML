import os
import cv2 as cv
import numpy as np

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']

# p = []
# for i in os.listdir(r'C:\Users\User\Desktop\python_in_vs\Learning Machine Learning\OpenCV\Faces\train'):
#     p.append(i)

# print(p)

DIR = r'C:\Users\User\Desktop\python_in_vs\Learning Machine Learning\OpenCV\Faces\train'

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('Training has finished!')

features = np.array(features, dtype='object')
labels = np.array(labels)


# print(f'Length of the features :  {len(features)}')
# print(f'Length of the features :  {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

face_recognizer.save('OpenCV/face_trained_model.yml')

np.save('OpenCV/features.npy', features)
np.save('OpenCV/labels.npy', labels)
