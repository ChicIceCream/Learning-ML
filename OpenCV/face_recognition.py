import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = npl.load('OpenCV/features.npy')
labels = np.load('OpenCV/labels.npy')
