# Week 2

Previously, we saw that we could use machine learning, data, and labels, and have a computer figure out the rules. This week we will go into applying this concept to computer vision.

## An Introduction to Computer Vision

- Computer vision is the field of having a computer understand and label what is present in an image.
- We will look at how to train a computer with the Fashion MNIST dataset (contains 70K images spread acorss 10 different items of clothing) which will then allow the computer to identify similar pieces of clothing

## Writing Code to Load Training Data

- Fashion MNIST is available as a dataset with an API call in TensorFlow:

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

- In the Fashion MNIST dataset, 60K images will be used to train the model and 10K images (images not previously seen) will be used to test the model

## Coding a Computer Vision Neural Network

- Below is an example of a neural network with multiple layers, where the middle layer acts as the hidden layer with 128 neurons:

```python
model = keras.Sequential([
  keras.layers.Flatten(input_shape = (28, 28)),
  keras.layers.Dense(128, activiation = tf.nn.relu),
  keras.layers.Dense(10, activation = tf.nn.softmax)
])
```

- We now have 10 neurons in the model since we have 10 classes of clothing in the dataset
- Since our images are 28 x 28, the line `keras.layers.Flatten(input_shape = (28, 28))` flattens those pixels into a linear array

## Using Callbacks to Control Training

- How can we use callbacks to stop training when desired loss/accuracy is reached? Let's examine the following:

```python
# Initialize imports
import tensorflow as tf
print(tf.__version__)

# Load training data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize data
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Define model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = tf.nn.relu),
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# Compile and train the model
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 5)

# Evaluate model
model.evaluate(test_images, test_labels)
```

- Where we define `model.fit(training_images, training_labels, epochs = 5)` we can write a callback to monitor loss and cancel when loss is acceptable:

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if (logs.get('loss') < 0.4):
      print("\nLoss is low so cancelling training!")
      self.model.stop_training = True
```

- Now we have the callback we can change add the following to the original code:

```python
# Initialize callback

callbacks = myCallback()

# ...

# Compile and train the model
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 5, callbacks = [callbacks])
```
