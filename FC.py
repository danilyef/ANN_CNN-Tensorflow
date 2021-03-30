import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from layers import Dense_layer, Softmax_layer, Dropout_layer, Batch_Normalization

#Loading dataset and preprocessing:
(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()
unique_labels = set(Y_train)

X_train = X_train/255 
X_test = X_test/255 

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)

X_train= X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = Y_train.astype('int32')
Y_test = Y_test.astype('int32')

dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dataset_train = dataset_train.shuffle(buffer_size=1024).batch(64)

dataset_test  = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
dataset_test = dataset_test.shuffle(buffer_size=1024).batch(64)


#model creation 
class ANN_Model(Model):

    def __init__(self):
        super(ANN_Model,self).__init__()

        self.dense_layer_1 = Dense_layer(784,512)
        self.batch_norm_1 = Batch_Normalization(convolution = False,depth = 512,decay = 0.95)

        self.dense_layer_2 = Dense_layer(512,256)
        self.batch_norm_2 = Batch_Normalization(convolution = False,depth = 256,decay = 0.95)
 
        self.dense_layer_3 = Dense_layer(256,128)
        self.dropout_layer_1 = Dropout_layer(0.3) 

        self.softmax = Softmax_layer(128,len(unique_labels))

    def call(self,x,training=None):
        x = self.dense_layer_1(x)
        x = self.batch_norm_1(x,training = training)
        x = tf.nn.relu(x)

        x = self.dense_layer_2(x) 
        x = self.batch_norm_2(x,training = training)
        x = tf.nn.relu(x)


        x = self.dense_layer_3(x)
        x = self.dropout_layer_1(x,training = training)
        x = tf.nn.relu(x) 

        x = self.softmax(x)

        return x 

model= ANN_Model()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#training
@tf.function
def training(X,y):
	with tf.GradientTape() as tape:
		predictions = model(X,training = True)
		loss = loss_object(y,predictions)
	gradients = tape.gradient(loss,model.trainable_variables)
	optimizer.apply_gradients(zip(gradients,model.trainable_variables))
	train_loss(loss)
	train_accuracy(y, predictions)


@tf.function
def testing(X,y):
	predictions = model(X,training = False)
	loss = loss_object(y,predictions)
	test_loss(loss)
	test_accuracy(y,predictions)



EPOCHS = 20

for epoch in range(EPOCHS):
	for X,y in dataset_train:
		training(X,y)

	for X,y in dataset_test:
		testing(X,y)

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()


