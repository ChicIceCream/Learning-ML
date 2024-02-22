import cv2 as cv

img = cv.imread('OpenCV/Photos/lady.jpg')
cv.imshow('Person_Lady', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Load the Haar cascade classifier XML file
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the classifier was loaded successfully
if haar_cascade.empty():
    print("Error: Unable to load the Haar cascade classifier XML file.")
else:
    print("Haar cascade classifier XML file loaded successfully.")

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

cv.waitKey(0)
