# How to start your first AI project
In this article, you'll learn to build a convolutional neural network (CNN) and train it with the [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist). Fashion MNIST Dataset is a data with 70,000 images and contains 10 classes of clothing with a dimension of 28 by 28 grayscale image color.

This post is also available as a [Nimblebox notebook](https://github.com/NimbleBoxAI/how-to-start-your-first-ai-project). Feel free to copy and run the notebook in your account and mess around with the code. To learn more about Nimblebox, read the official [blog post]().

## Data Exploration and Training Model

### Fashion MNIST Dataset
The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples where each example is a 28 x 28 grayscale image, associated with a label from 10 classes to classify. These 10 classes are:

| Label | Description |
|:-----:|:-----------:|
|   0   | T-shirt/top |
|   1   | Trouser |
|   2   | Pullover |
|   3   | Dress |
|   4   | Coat |
|   5   | Sandal |
|   6   | Shirt |
|   7   | Sneaker |
|   8   | Bag |
|   9   | Ankle boot |

Here's an example how the data looks (_each class takes three-rows_):

![Fashion MNIST Dataset](helper_images/fashion-mnist.png)

Now that we know what our dataset has, let’s jump to the code.

### Loading Fashion MNIST Dataset
First, let's check our [TensorFlow](www.tensorflow.org) version and import tensorflow. Then we download fashion-mnist.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check tensorflow version
print("Tensorflow version = ", tf.__version__)

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
```

### Data Visualization
Now that we have downloaded our dataset, it is always a good idea to visualize our data if possible.

```python
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = int(np.random.randint(0, 59999, 1))

# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]

# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))

# Show one of the images from the training dataset
plt.imshow(x_train[img_index])
```

### Data Normalization
We then normalize the data dimensions so that they are of approximately the same scale.

```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
```

### Split the data into train/validation/test data sets and One-hot encoding the labels
In the earlier step, we divided our dataset into 60,000 datasets for training and 10,000 test datasets. Now we further split the training data into train/validation. Here is how each type of dateset is used in deep learning:
* Training data — used for training the model
* Validation data — used for tuning the hyperparameters and evaluate the models
* Test data — used to test the model after the model has gone through initial vetting by the validation set.

We will also one-hot encode our labels.

```python
# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print the shape of training, validation, and test datasets
print("x_train shape:", x_train.shape, "|”, "y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape, "|”, "y_valid shape:", y_valid.shape)
print("x_test shape:", x_test.shape, "|”, "y_test shape:", y_test.shape)
```

### Create Model Architecture
We will use Keras Functional model API to create a simple CNN model repeating a few layers of a convolution layer followed by a pooling layer then a dropout layer.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.models import Model

inputs = Input(shape=(28, 28, 1))

layer_1 = Conv2D(filters=128, kernel_size=2, padding='same', activation='relu')(inputs)
layer_1 = MaxPooling2D(pool_size=2)(layer_1)
layer_1 = Dropout(0.3)(layer_1)

layer_2 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(layer_1)
layer_2 = MaxPooling2D(pool_size=2)(layer_2)
layer_2 = Dropout(0.3)(layer_2)

layer_3 = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(layer_2)
layer_3 = MaxPooling2D(pool_size=2)(layer_3)
layer_3 = Dropout(0.3)(layer_3)

final_layer = Flatten()(layer_3)
final_layer = Dense(256, activation='relu')(final_layer)
final_layer = Dropout(0.5)(final_layer)
final_layer = Dense(10, activation='softmax')(final_layer)

model = Model(inputs=inputs, outputs=final_layer)

# summarize layers
print(model.summary())

# plot graph
tf.keras.utils.plot_model(model, to_file='helper_images/fashion_mnist_cnn.png')
```
![Model Visualization](helper_images/fashion_mnist_cnn.png)

### Compile our Model
We use model.compile() to configure the learning process before training the model by defining the type of loss function, optimizer, and the metrics evaluated by the model during training and testing.

```python
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
```

### Training our Model
We will train the model with a batch_size of 64 and 30 epochs and save our model only when the validation accuracy improves using ModelCheckpoint API()

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath="models/model_weights_best.hdf5", verbose = 1, save_best_only=True)

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=30,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])
```

### Load Model and Test Accuracy
Now that we have trained our model, let’s load our best model and test our accuracy.

```python
# Load the weights with the best validation accuracy
model.load_weights("models/model_weights_best.hdf5")

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
```

## Visualize the predictions
Now let's visualize the prediction using the model you just trained. If the prediction matches the true label, the title will be green; otherwise, it's displayed in red.

```python
y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
```
![Visualization of Prediction](helper_images/final-classification-fashion-mnist.png)


## Congratulations!
We have successfully trained a CNN to classify fashion-MNIST with near 90% accuracy.
