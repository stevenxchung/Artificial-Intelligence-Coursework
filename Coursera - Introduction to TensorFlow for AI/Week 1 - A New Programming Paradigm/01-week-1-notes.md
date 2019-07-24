# Week 1 - A New Programming Paradigm

In this first week we go over the basics of machine learning and TensorFlow.

## A Primer in Machine Learning

- Traditional programming:
  - Input:
    - Rules
    - Data
  - Output: answers
- Machine learning:
  - Input:
    - Answers
    - Data
  - Output: rules

## The _Hello World_ of Neural Networks

- Machine learning is all about a computer learning the patterns that distinguishes things
- We can write a simple _Hello World_ model for a neural network with one neuron (the most basic neural network) with the following:

```python
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
```

- `keras` makes it easy to define neural networks
- `Dense` defines a layer of connected neurons, in the model shown above, there is only one layer with one unit
- `Sequential` denotes successive layers of the neural network in a sequence

```python
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
```

- `loss` function measures how good or bad each guess is, in the example above, the loss function is represented by the mean squared error
- `optimizer` function thinks about how good or bad the guess was using the loss function, in the example above, the optimizer is represented by the stochastic gradient descent

- We can train the model by adding the _X_ and _Y_ data arrays to the code and `fit` them to the model
- `epochs` is used to denote the number of times we go through the training loop
- `predict` is used to give the approximate value based on the trained model

```python
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(xs, ys, epochs = 500)

print(model.predict([10, 0]))
```
