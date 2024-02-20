import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

blank = np.zeros((500,500,3), dtype='uint8')
# cv.imshow('Blank Image', blank)

# img = cv.imread("OpenCV\Photos\cat.jpg")
# resized_img = rescaleFrame(img)
# cv.imshow('Cat', resized_img)

blank[200:300, 300:400] = 0,255,0
cv.imshow('Green Screen', blank)

cv.waitKey(0)