import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and 
    softmax layer.

    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):
        """
        
        input
        ------------
        sequence_length: type_int
             The length of our sentences. Remember that we padded all our
             sentences to have the same length (59 for our
             data set).
        num_classes: type_int
            Number of classes in the output layer, two in our case (positive 
            and negative).
        vocab_size: type_int
            The size of our vocabulary. This is needed to define the size of
            our embedding layer, which will have shape 
            [vocabulary_size, embedding_size].
        embedding_size: type_int
            The dimensionality of our embeddings
        filter_sizes: array_type
            he number of words we want our convolutional filters to cover. We
            will have num_filters for each size specified here. For example, 
            [3, 4, 5] means that we will have filters that slide over 3, 4 
            and 5 words respectively, for a total of 3 * num_filters filters.
        num_filters: int_type
            The number of filters per filter size (see above).

        returns: CNN
        """
        # Implementation
        # 2nd arg is shape of input vector
        # None allows for arbitrary batch sizes
        self.input_x = tf.placeholder(tf.int32,
                                      [None, sequence_length],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, num_classes],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")
        
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # Name scope helps when visualizing in TensorBoard
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolutional Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, 
                        stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), 
                        name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID", # VALID - Dont pad the edges
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h, 
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pooled)

            # Combined all pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(3, pooled_outputs)
            # using -1 flattens the dimensions
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                            self.dropout_keep_prob)

            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([num_filters_total,
                    num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]),
                    name='b')
                self.scores = tf.nn.xw_plus_b(self.h_drop,
                                              W,
                                              b,
                                              name='scores')
                self.predictions = tf.argmax(self.scores, 1,
                                             name='predictions')
            # Calculate mean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,
                        self.input_y)
                self.loss = tf.reduce_mean(losses)

            # Calculate accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions,
                    tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                    'float'), name='accuracy')


