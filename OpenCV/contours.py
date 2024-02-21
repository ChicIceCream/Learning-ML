import cv2 as cv
import numpy as np

img = cv.imread('OpenCV\Photos\cat.jpg')

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleFrame(img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayed img', gray)

# blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# canny = cv.Canny(blur, 125, 175)
# cv.imshow("Canny", canny)

ret, thresh = cv.threshold(gray, 125, 125, cv.THRESH_BINARY)
cv.imshow('Threshed', thresh)

contours, heirarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!!!')

cv.drawContours(blank, contours, -1, (0,0,255), thickness=1)
cv.imshow('Contoured', blank)

cv.waitKey(0)