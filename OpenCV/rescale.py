import cv2 as cv

def rescaleFrame(frame, scale=0.1): # ! will work for images, video, and live capture
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height): # ! Only for live videos 
    capture.set(3, width)
    capture.set(4, height)

img = cv.imread('OpenCV\Photos\cat.jpg')

cv.imshow("Cat", img)

resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)


capture = cv.VideoCapture('OpenCV\Videos\dog.mp4') #! put 0 for laptop camera

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)
    
    cv.imshow('Video of dog', frame)
    cv.imshow("Video Resized", frame_resized)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
