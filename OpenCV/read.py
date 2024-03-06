import cv2 as cv

capture = cv.VideoCapture(0) #! put 0 for laptop camera

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video of dog', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

# img = cv.imread('OpenCV\Photos\cat.jpg')

# cv.imshow("Cat", img)


cv.waitKey(0)