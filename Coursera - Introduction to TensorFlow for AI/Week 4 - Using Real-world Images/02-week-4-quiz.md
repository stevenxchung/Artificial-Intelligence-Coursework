# Week 4 Quiz

1. Using Image Generator, how do you label images? Answer: _It's based on the directory the image is contained in_

2. What method on the Image Generator is used to normalize the image? Answer: `rescale`

3. How did we specify the training size for the images? Answer: _The `target_size` parameter on the training generator_

4. When we specify the input_shape to be (300, 300, 3), what does that mean? Answer: _Every image will be 300 x 300 pixels, with 3 bytes to define color_

5. If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here? Answer: _You're overfitting on your training data_

6. Convolutional Neural Networks are better for classifying images like horses and humans because: _All of the above_

7. After reducing the size of the images, the training results were different. Why? Answer: _We removed some convolutions to handle the smaller images_
