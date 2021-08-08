import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_eager_execution() 
import os 

import matplotlib.pyplot as plt 
plt.style.use('seaborn')
from maf import MAF

tfd = tfp.distributions
tfb = tfp.bijectors

class RealNVP:
    def __init__(self, dtype, tf_version, batch_size, params, hidden_units, base_dist, dims, shift_only, is_constant_jacobian, masked_dimension_count):
        self.tf_version = tf_version
        self.dtype = dtype
        self.base_dist = base_dist
        self.dims = dims
        self.params = params 
        self.hidden_units = hidden_units 
        self.batch_size = batch_size
        self.shift_only = shift_only
        self.is_constant_jacobian = is_constant_jacobian
        self.masked_dimension_count = masked_dimension_count
        
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
    
    def override_masked_dimension_count(self, new_dim):
        self.masked_dimension_count = new_dim
    
    def get_shift_scale_func(self, data):
        func = tfb.real_nvp_default_template(self.hidden_units, self.shift_only)
        return func
    
    def make_realnvp(self,data):
        distribution = self.base_dist
        sample_shape = self.get_dims(data)
        shift_scale_function = self.get_shift_scale_func(data)
        bijector = tfb.RealNVP(num_masked=self.masked_dimension_count, shift_and_log_scale_fn=shift_scale_function, is_constant_jacobian=self.is_constant_jacobian)
        realnvp = tfd.TransformedDistribution(tfd.Sample(distribution, sample_shape), bijector)
        return realnvp