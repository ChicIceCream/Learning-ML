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

# 1. Paint whole screen a colour
blank[:] = 0,255,0
cv.imshow('Green Screen', blank)

# 2. Only painitng a little place
blank[200:300, 300:400] = 0,0,255
cv.imshow('Green Screen with squre', blank)

# 3. Drawing a rectangle
cv.rectangle(blank, (0,0), (250,500), (255,0,0), thickness=-1) #! use thinkness=-1 to fill whole
cv.imshow('Rectanlge', blank)

# 4. drawing a circle
cv.circle(blank, (250,250), 40, (255,255,255), thickness=4)
cv.imshow('Cricle', blank)

# 5. Drawing a line
cv.line(blank, (50,250), (450,250), (0,0,0) , thickness=6)
cv.imshow("Line", blank)

# 6. Wrtiting text in the image
cv.putText(blank, 'Hallooo', (100,100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (112,124,9), thickness=4)
cv.imshow('Text', blank)

cv.waitKey(0) 