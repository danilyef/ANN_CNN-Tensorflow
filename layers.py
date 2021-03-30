import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


class Convolution_Layer(tf.keras.layers.Layer):
    
    def __init__(self,kernel_height,kernel_width,channel_in,channel_out,stride,padding):
        super(Convolution_Layer,self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.initializer = tf.initializers.GlorotUniform()
        
        #weights:
        self.W = tf.Variable(self.initializer(shape = (kernel_height,kernel_width,channel_in,channel_out)))
        self.b = tf.Variable(self.initializer(shape = (channel_out,)))
        
    
    def call(self,x):
        x = tf.nn.conv2d(x,self.W,strides = self.stride,padding = self.padding) + self.b
        return x
    
    
class Batch_Normalization(tf.keras.layers.Layer):
    def __init__(self,depth,decay,convolution):
        super(Batch_Normalization,self).__init__()
        self.mean = tf.Variable(tf.constant(0.0,shape = [depth]),trainable = False)
        self.var = tf.Variable(tf.constant(1.0,shape = [depth]),trainable = False)
        self.beta = tf.Variable(tf.constant(0.0,shape = [depth]))
        self.gamma = tf.Variable(tf.constant(1.0,shape = [depth]))
        #exponentiall moving average object
        self.mov_avg = tf.train.ExponentialMovingAverage(decay = decay)
        self.epsilon = 0.001
        self.convolution = convolution
        
    def call(self,x,training = True):
        
        if training:
            if self.convolution:
                batch_mean,batch_var = tf.nn.moments(x, axes=[0, 1, 2],keepdims = False)
            else:
                batch_mean,batch_var = tf.nn.moments(x, axes=[0],keepdims = False)

            as_mean = self.mean.assign(batch_mean)
            as_variance = self.var.assign(batch_var)
            #ensured argument to be evaluated before anything you define in the with block
            with tf.control_dependencies([as_mean,as_variance]):
                self.mov_avg.apply([self.mean,self.var])
                x = tf.nn.batch_normalization(x = x,mean = batch_mean,variance = batch_var,offset = self.beta,scale = self.gamma,variance_epsilon = self.epsilon)
                
        else:
            mean = self.mov_avg.average(self.mean)
            var = self.mov_avg.average(self.var)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            x = tf.nn.batch_normalization(x,mean,var,local_beta,local_gamma,self.epsilon)
            
        return x
            


class MaxPool(tf.keras.layers.Layer):
    def __init__(self,kernel_size,strides,padding):
        super(MaxPool,self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        
    def call(self,x):
        return tf.nn.max_pool2d(x,ksize = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.strides,self.strides,1],padding = self.padding)



class Dense_layer(tf.keras.layers.Layer):
    def __init__(self,dim_in,dim_out):
        super(Dense_layer,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        
        #weights:
        self.W = tf.Variable(self.initializer(shape =(dim_in,dim_out)))
        self.b = tf.Variable(self.initializer(shape =(dim_out,)))
        
        
    def call(self,x):
        return x @ self.W + self.b


class Flatten_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Flatten_layer,self).__init__()
        
    def call(self,x):
        return tf.reshape(x, [x.shape[0],-1])


class Softmax_layer(tf.keras.layers.Layer):
    def __init__(self,dim_in,dim_out):
        super(Softmax_layer,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        
        #weights:
        self.W = tf.Variable(self.initializer(shape =(dim_in,dim_out)))
        self.b = tf.Variable(self.initializer(shape =(dim_out,)))

    def call(self,x):
        return tf.nn.softmax(tf.matmul(x,self.W) + self.b)



class AvgPooling(tf.keras.layers.Layer):
    def __init__(self,kernel_height,kernel_width,strides,padding):
        super(AvgPooling,self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.strides = strides
        self.padding = padding
        
        
    def call(self,x):
        return tf.nn.avg_pool(input = x,ksize = [self.kernel_height,self.kernel_width],strides = self.strides,padding = self.padding)



class Dropout_layer(Layer):
	def __init__(self,rate):
		super(Dropout_layer,self).__init__()
		self.rate = rate

	def call(self,x,training = None):
		if training:
			return tf.nn.dropout(x,self.rate)
		return x