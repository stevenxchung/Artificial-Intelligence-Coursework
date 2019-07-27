# Week 3 - Enhancing Vision with Convolutional Neural Networks

Previously, we created a neural network (deep neural network) to pattern match a set of fashion items to labels. In this lesson we will go over how we can use convolutional neural networks to reduce wasted space in each image and focus on the pixels with valuable information.

## What are Convolutions and Pooling?

- A convolution involves having a filter and passing that filter over the image to change the underlying image
- Pooling is a way to compress an image, typically the computer can go over the image four pixels at a time and pick the biggest value of the four (see lecture for diagram)

## Implementing Convolutional Layers

- We can implement convolutional layers in code as follows:

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dense(10, activation = 'softmax')
])
```

- In the first layer above, we generate 64 filters which are 3 x 3 where each result which is negative is thrown away

## Implementing Pooling Layers

- The second layer is where pooling is implemented, max-pooling takes the maximum value of a 2 x 2 pool
- There is another convolution and pooling layer after the first two in the code shown previously such that, by the time the model reaches the flattening step the content has been greatly reduced
- We can use `model.summary()` to inspect the layers of the model and see the journey of the image through the convolutions
