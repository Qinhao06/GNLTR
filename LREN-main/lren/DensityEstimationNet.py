# -*- coding: utf-8 -*-
import tensorflow as tf

class DensityEstimationNet:
    def __init__(self, layer_sizes, activation=tf.nn.relu):

        self.layer_sizes = layer_sizes
        self.activation = activation

    def inference(self, z, dropout_ratio=None):

        with tf.compat.v1.variable_scope("DensityEstimationNet"):
            N_layer = 0
            for size in self.layer_sizes[:-1]:
                N_layer += 1
                z = tf.compat.v1.layers.dense(z, size, activation=self.activation,
                    name="Lay_{}".format(N_layer))
                if dropout_ratio is not None:
                    z = tf.compat.v1.layers.dropout(z, dropout_ratio,
                        name="Drop_Ratio_{}".format(N_layer))

            size = self.layer_sizes[-1]
            logits = tf.compat.v1.layers.dense(z, size, activation=None, name="logits")
            output = tf.compat.v1.nn.softmax(logits)
            # output = tf.compat.v1.layers.softmax(logits)
        return output
