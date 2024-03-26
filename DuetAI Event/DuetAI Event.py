import cv2
import numpy as np

# Load the image
image = cv2.imread('DuetAI Event\image.jpg')

# Define the coordinates and size of the first square
x1, y1, width1, height1 = np.random.randint(0, image.shape[1] - 100), np.random.randint(0, image.shape[0] - 100), 100, 100

# Define the coordinates and size of the second square
x2, y2, width2, height2 = np.random.randint(0, image.shape[1] - 100), np.random.randint(0, image.shape[0] - 100), 100, 100

# Draw the first square
cv2.rectangle(image, (x1, y1), (x1 + width1, y1 + height1), (0, 255, 255), 2)

# Draw the second square
cv2.rectangle(image, (x2, y2), (x2 + width2, y2 + height2), (0, 255, 255), 2)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()