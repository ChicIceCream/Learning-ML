import cv2 as cv

def rescaleFrame(frame, scale=1.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('OpenCV/Photos/group 1.jpg')
# cv.imshow('group', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# Load the Haar cascade classifier XML file
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the classifier was loaded successfully
if haar_cascade.empty():
    print("Error: Unable to load the Haar cascade classifier XML file.")
else:
    print("Haar cascade classifier XML file loaded successfully.")

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=1)

img = rescaleFrame(img)
cv.imshow('Detected Faces', img)

cv.waitKey(0)
