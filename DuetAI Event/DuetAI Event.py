import cv2
import numpy as np

# Load the image
image = cv2.imread('DuetAI Event\image.jpg')

# Define the coordinates and size of the first square
x1, y1, width1, height1 = 100, 100, 50, 50

# Define the coordinates and size of the second square
x2, y2, width2, height2 = 200, 200, 70, 70

# Draw the first square
cv2.rectangle(image, (x1, y1), (x1 + width1, y1 + height1), (0, 0, 255), -1)
cv2.putText(image, "Me!!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Draw the second square
cv2.rectangle(image, (x2, y2), (x2 + width2, y2 + height2), (0, 0, 255), -1)
cv2.putText(image, "Prashant Sir", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()