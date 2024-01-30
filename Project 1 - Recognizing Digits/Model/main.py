import cv2 as cv    
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  

model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

# print(len(x_test))
# print(len(y_test))

model.save('Project 1 - Recognizing Digits\digits.model')

for x in range(1,5):
    img = cv.imread(f'Project 1 - Recognizing Digits\{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    predication = model.predict(img)
    print(f'The prediction based on my knowledge is that this image is : {(np.argmax(predication))}')
    plt.imshow(img[0], cmap="binary")
    plt.show()