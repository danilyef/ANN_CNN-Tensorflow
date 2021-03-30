# Standard Convolutional and Fully Connected neural networks using Tensorflow 2.4 (low level API)

### Fully Connected neural network:

1. Network consists of 3 fully connected layers, where each neural of layer ***t*** is connected to each neuron in the layer ***t+1***.
2. In order to speed up learning and make network more stable batch normalization technique was used: normalizing each batch of the neural network using batch mean and variance. Fot the test set we are using moving average mean and variance, which are calculated during training procedure.
3. As reqularization technique we have used dropout: each neuron in the layer can be turned off with a probability ***p***.
4. For prediction we are using softmax function.

### Convolutional neural network:

When images are input objects, we have some information about the structure of the object(location of particular pixels to each other). Fully connected network doesn't use this information. Convolutional neural network uses convolutional operation(cross-correlation to be precise) instead of matrix multiplication. This operation takes into account local structure of the object.

The main idea of CNNs is to use local receptive fields which are applied on all positions of the image by sharing parameters across neurons. Plus to this we are using pooling operations that are great for detecting features of the images at different levels

1. Network consists of 4 convolutional layers.
2. As pooling technique we are using max pool.
3. As reqularization technique we have used dropout: each neuron in the layer can be turned off with a probability ***p***.
4. After main CNN part we flatten the last convolutional layer and apply basic fully connected layer with dropout.
5. For prediction we are using softmax function.



##### Dataset:

We will be using:
1. Fashion-MNIST dataset for FC Network.
2. Cifar-10 dataset for CNN Network.



