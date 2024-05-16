import time
import mediapipe as mp
import cv2 
import numpy
import HandTrackingModule as htm

####################
wCam, hCam = 720, 480
####################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, wCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


while True:
    success, img = cap.read()
    
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        
        
        cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), cv2.FILLED)
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS : {(int(fps))}', (10, 70), cv2.FONT_HERSHEY_COMPLEX,
                2, (255, 0, 0), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
