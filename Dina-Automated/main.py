import cv2
import cvzone
from cvzone.FPS import FPS
import numpy as np
import pyautogui
from mss import mss

def caputure_screen_region(x, y, desired_width, desired_height):
    screenshot = pyautogui.screenshot(region=(x, y, desired_width, desired_height))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    
    return screenshot

while True:
    imgGame = caputure_screen_region(0, 0, 500, 500)
    cv2.imshow("Game", imgGame)
    cv2.waitKey(1)  