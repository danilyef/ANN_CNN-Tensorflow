from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from layers import Dense_layer, Softmax_layer, Dropout_layer, Batch_Normalization, Convolution_Layer, MaxPool, Flatten_layer
import tensorflow as tf
import numpy as np
#Preprocessing:
(X_train,Y_train),(X_test,Y_test) = datasets.cifar10.load_data()
X_train = X_train /255
X_test = X_test /255 

unique_labels = np.arange(0,np.max(Y_train,axis= 0) + 1)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Y_train = to_categorical(Y_train,len(unique_labels))
#Y_test = to_categorical(Y_test,len(unique_labels))


dataset_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train))#we get the slices of an array in the form of objects by using 
#creates a dataset with a separate element for each row of the input tensor:
# = tf.constant([[1, 2], [3, 4]])
#ds = tf.data.Dataset.from_tensor_slices(t)   # [1, 2], [3, 4]
dataset_train = dataset_train.shuffle(buffer_size = 1024).batch(64)

dataset_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
dataset_test = dataset_test.shuffle(buffer_size = 1024).batch(64)


class CNN_Model(Layer):

    def __init__(self):
        super(CNN_Model,self).__init__()
        self.conv_1 = Convolution_Layer(kernel_height = 3,kernel_width = 3,channel_in = 3,channel_out = 32,stride = 1,padding = 'SAME')
        self.batch_norm_1 = Batch_Normalization(depth = 32,decay = 0.99,convolution = True)

        self.conv_2= Convolution_Layer(kernel_height = 3,kernel_width = 3,channel_in = 32,channel_out = 32,stride = 1,padding = 'VALID')
        self.batch_norm_2 = Batch_Normalization(depth = 32,decay = 0.99,convolution = True)
        
        self.max_pool_1 = MaxPool(kernel_size = 2,strides = 1,padding = 'VALID')
        
        self.conv_3 = Convolution_Layer(kernel_height = 3,kernel_width = 3,channel_in = 32,channel_out = 64,stride = 1,padding = 'SAME')
        self.batch_norm_3 = Batch_Normalization(depth = 64,decay = 0.99,convolution = True)

        self.conv_4 = Convolution_Layer(kernel_height = 3,kernel_width = 3,channel_in = 64,channel_out = 64,stride = 1,padding = 'VALID')
        self.batch_norm_4 = Batch_Normalization(depth = 64,decay = 0.99,convolution = True)
        
        self.max_pool_2 = MaxPool(kernel_size = 2,strides = 1,padding = 'VALID')



        self.flatten = Flatten_layer()
        self.dense_1 = Dense_layer(43264,512)
        self.dropout_1 = Dropout_layer(0.5)
        self.softmax = Softmax_layer(512,len(unique_labels))
        
        
        


    def call(self,inputs,training = None):
        X = self.conv_1(inputs)
        X = self.batch_norm_1(X,training)
        X = tf.nn.relu(X)
        
        X = self.conv_2(X)
        X = self.batch_norm_2(X,training)
        X = tf.nn.relu(X)
        X = self.max_pool_1(X)
        
        X = self.conv_3(X)
        X = self.batch_norm_3(X,training)
        X = tf.nn.relu(X)
        
        X = self.conv_4(X)
        X = self.batch_norm_4(X,training)
        X = tf.nn.relu(X)
        X = self.max_pool_2(X)


        X = self.flatten(X)
        X = self.dense_1(X)
        X = tf.nn.relu(X)
        X = self.dropout_1(X,training)
        X = self.softmax(X)
        return X


model = CNN_Model()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()#used in backprop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')#mean of the losses per observation
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def training(X,y):
	with tf.GradientTape() as tape:#Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
		predictions = model(X,training = True)
		loss = loss_object(y,predictions)

	gradients = tape.gradient(loss,model.trainable_variables)
	optimizer.apply_gradients(zip(gradients,model.trainable_variables))
	train_loss(loss) 
	train_accuracy(y,predictions)



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