import tensorflow as tf 
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


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

model.fit(train_images, train_labels, epochs=3)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)


