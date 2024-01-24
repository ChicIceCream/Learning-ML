import cv2 as cv    
import numpy as np
import matplotlib as plt
import tensorflow as tf

mnist = tf.keras.dataset.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.Layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.Layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.Layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.Layers.Dense(10, activation=tf.nn.softmax))  

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')