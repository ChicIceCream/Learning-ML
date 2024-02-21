import cv2 as cv

img = cv.imread('OpenCV\Photos\cat.jpg')

def rescaleFrame(frame, scale=0.1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayed img', gray)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny", canny)

cv.waitKey(0)