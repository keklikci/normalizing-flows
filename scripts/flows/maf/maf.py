import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_eager_execution() 
import os 

import matplotlib.pyplot as plt 
plt.style.use('seaborn')

tfd = tfp.distributions
tfb = tfp.bijectors

class MAF:
    def __init__(self, dtype, tf_version, batch_size, params, hidden_units, base_dist, dims):
        self.tf_version = tf_version
        self.dtype = dtype
        self.base_dist = base_dist
        self.dims = dims
        self.params = params 
        self.hidden_units = hidden_units 
        self.batch_size = batch_size
        
    def get_tf_version(self):
        return self.tf_version
    
    def get_session(self):
        return tf.compat.v1.Session()
        
    def get_dims(self, data):
        return data.shape[1]
    
    def create_tensor(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data.astype(self.dtype))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=data.shape[0])
        dataset = dataset.prefetch(2*self.batch_size)
        dataset = dataset.batch(self.batch_size)
        data_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        samples = data_iterator.get_next()
        return samples
    
    def get_shift_scale_func(self, data):
        func = tfb.AutoregressiveNetwork(params=self.params, hidden_units=self.hidden_units)
        return func 
    
    def make_maf(self, data):
        distribution = self.base_dist
        sample_shape = self.get_dims(data)
        shift_scale_function = self.get_shift_scale_func(data)
        bijector = tfb.MaskedAutoregressiveFlow(shift_scale_function)
        maf = tfd.TransformedDistribution(tfd.Sample(distribution, sample_shape), bijector)
        return maf
 