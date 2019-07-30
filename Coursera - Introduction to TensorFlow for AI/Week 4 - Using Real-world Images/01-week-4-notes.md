# Week 4 - Using Real-world Images

Previously, we learned how we could use CNN (convolutional neural networks) to improve computer vision. However, these techniques were limited in that we only looked at 28 x 28 images which were centered. This week we will see how we can apply the core idea of CNN to more complex problems.

## Understanding ImageGenerator

- We can use ImageGenerator in TensorFlow to point it at a sub-directory and then the sub-directors of that will automatically generate labels
- We can import ImageGenerator as follows:

```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
```

- Furthermore, we can use instantiate an image generator like so:

```python
train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size = (300, 300),
  batch_size = 128,
  class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size = (300, 300),
  batch_size = 32,
  class_mode = 'binary')
```

- Above, we observe that the directory of the images is given in the first argument of `test_datagen.flow_from_directory()`
- `target_size` resizes the images as they are loaded
- `batch_size` specifies the size of the batch of images during loading
- `class_mode` allows us to specify if the class of data is binary (e.g., horse vs humans) or another class

## Definding a ConvNet to use Complex Images

- A ConvNet is a model that has more than one CNN applied to it, below is an example:

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation = 'relu'),
  tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```

- In the above code, the output layer has changed from `softmax` (one neuron per class) to `sigmoid` (one neuron for two classes) which is great for binary classification
