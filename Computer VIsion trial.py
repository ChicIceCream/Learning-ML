import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()


plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

model.fit(training_images, training_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print(test_loss, test_accuracy)

# classifications = model.predict(test_images)
# print(classifications[0])  

