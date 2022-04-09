import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



class ChannelAttention(tf.keras.layers.Layer):
    """
    ...
    """


    def __init__(self, filters, ratio, **kwargs):

        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio


        def build(self, input_shape):

            self.shared_layer_one = tf.keras.layers.Dense(self.filters//self.ratio,
                                                        activation='relu', kernel_initializer='he_normal', 
                                                        use_bias=True, 
                                                        bias_initializer='zeros')

            self.shared_layer_two = tf.keras.layers.Dense(self.filters,
                                                        kernel_initializer='he_normal',
                                                        use_bias=True,
                                                        bias_initializer='zeros')


        def call(self, inputs):

            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            avg_pool = self.shared_layer_one(avg_pool)
            avg_pool = self.shared_layer_two(avg_pool)

            max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
            max_pool = tf.keras.layers.Reshape((1, 1, self.filters))(max_pool)

            max_pool = shared_layer_one(max_pool) #here
            max_pool = shared_layer_two(max_pool) #here

            attention = tf.keras.layers.Add()([avg_pool,max_pool])
            attention = tf.keras.layers.Activation('sigmoid')(attention)
            
            return tf.keras.layers.Multiply()([inputs, attention])


    def get_config(self):

        config = super().get_config()
        config.update({
            "filters": self.filters,
            "ratio": self.ratio,
        })
        return config   



class SpatialAttention(tf.keras.layers.Layer):
    """
    ...
    """


    def __init__(self, kernel_size, **kwargs):

        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
    

        def build(self, input_shape):

            self.conv2d = tf.keras.layers.Conv2D(filters = 1,
                                                kernel_size=self.kernel_size,
                                                strides=1,
                                                padding='same',
                                                activation='sigmoid',
                                                kernel_initializer='he_normal',
                                                use_bias=False)


        def call(self, inputs):
            
            avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
            max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
            attention = self.conv2d(attention)

            return tf.keras.layers.multiply([inputs, attention]) 


    def get_config(self):

        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config   



class StopOnPoint(tf.keras.callbacks.Callback):
    """
    ...
    """

    def __init__(self, point):

        super(StopOnPoint, self).__init__()
        self.point = point


    def on_train_batch_end(self, epoch, logs=None):

        accuracy = logs["acc"]
        if accuracy >= self.point:
            self.model.stop_training = True



def ReshapeLayer(x):
    """
    ...
    """

    shape = x.shape
    reshape = tf.keras.layers.Reshape((shape[1], shape[2]*shape[3]))(x)

    return reshape

