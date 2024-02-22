import cv2 as cv

def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread("OpenCV\Photos\cat.jpg") 
img = rescaleFrame(img)
cv.imshow('Catto', img)

cv.waitKey(0)