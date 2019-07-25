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
