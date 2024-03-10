import cv2 as cv

def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# img = rescaleFrame(cv.imread("Photos\cat.jpg"))

# cv.imshow('cat', img)

# cv.waitKey()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    cv.imshow('Frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()

