import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # False, 2, 0.5, 0.5 
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(f'{id, lm}')  Prints the x, y and z positions of the id landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f'{id, cx, cy}')
                if id == 0:
                    cv.circle(img, (cx,cy), 15, (255, 0, 255),  cv.FILLED)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)    
    pTime = cTime
    
    cv.putText(img, f'FPS : {(int(fps))}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    
    cv.imshow("Image", img)
    cv.waitKey(1)