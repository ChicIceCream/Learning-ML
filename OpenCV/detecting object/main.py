import numpy as np
import cv2 as cv

def rescaleFrame(frame, scale=0.6):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread(r'detecting object\soccer_practice.jpg', 0)
template = cv.imread(r'detecting object\ball.PNG', 0)
h,w = template.shape
# print(img)

methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED,
            cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()
    
    result = cv.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
        
    bottom_right = (location[0] + w, location[1] + h)    
    cv.rectangle(img2, location, bottom_right, 0, 5)
    cv.imshow('Match', rescaleFrame(img2))
    cv.waitKey(0)
