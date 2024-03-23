import cv2

# Load the pre-trained model
model = cv2.dnn.readNetFromTensorflow('path/to/your/model.pb')

# Load the input image
image = cv2.imread('path/to/your/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw a red square around each detected face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with the red squares
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()