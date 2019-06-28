import tensorflow as tf
from functools import reduce

class DNDT(tf.keras.layers.Layer):
    
    def __init__(self, num_outputs, num_cuts, num_leaves, temperature = 0.1):
        super(DNDT, self).__init__()
        self.temperature = temperature
        
        weights = self.add_weight(
                name = 'leaf_score',
                shape = (num_leaves, num_outputs), 
                initializer = 'uniform',
                trainable = True
        )
        self.leaf_score = weights
        
        self.num_cuts = num_cuts
        self.cuts_list = []
        
        for i in range(len(self.num_cuts)):
            weights = self.add_weight(
                    name = 'cut_{}'.format(i),
                    shape = (self.num_cuts[i], ), 
                    initializer = 'uniform',
                    trainable = True
            )
            self.cuts_list.append(weights)
        
    # math stuff
    def tf_kron_prod(self, a, b):
        res = tf.einsum('ij,ik->ijk', a, b)
        res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
        return res
    
    def tf_bin(self, x, cut_idx):
        D = self.cuts_list[cut_idx].get_shape().as_list()[0]
        W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
        
        self.cuts_list[cut_idx] = tf.sort(self.cuts_list[cut_idx])
        
        b = tf.cumsum(tf.concat([tf.constant(0.0, shape = [1]), -self.cuts_list[cut_idx]], 0))
        
        h = tf.matmul(x, W) + b
        res = tf.nn.softmax(h/self.temperature)
        
        return res
        
    def call(self, inputs):
        
        leaf = reduce(self.tf_kron_prod,
                     map(lambda i: self.tf_bin(inputs[:, i:i + 1], i), range(len(self.cuts_list)) ))
        
        return tf.matmul(leaf, self.leaf_score)
