import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import os 
import matplotlib.pyplot as plt 
plt.style.use('seaborn')

tfd = tfp.distributions
tfb = tfp.bijectors

# experimental -- inspired by 
# Variational Inference with Normalizing Flows, Rezende & Shakir
# do not use for this setting, distorts base distribution a lot, no convergence in the end 
def parametrize(b):
    b = tf.cast(b, tf.float32)
    return tf.math.log1p(tf.exp(b)).numpy()

def h(a, r):
    return 1 / (a + r)

def h_prime(a, r):
    return -1 / (a + r)**2

class RadialFlow(tfb.Bijector):
    def __init__(self, a, b, x0, validate_args=True, name='radial-flow'):
        self.a = tf.cast(a, tf.float32) 
        self.b = tf.cast(b, tf.float32)
        self.x0 = tf.cast(x0, tf.float32)
        super(RadialFlow, self).__init__(validate_args=validate_args, 
                                         forward_min_event_ndims=0, 
                                         name=name)
        
        if validate_args:
            assert tf.math.greater_equal(self.b, -self.a).numpy() == True
    
    def _forward(self, x):
        r = tf.abs(x - self.x0)
        zhat = (x - self.x0) / r
        y = self.x0 + r*zhat + r * zhat * self.b * h(self.a, r)
        return y 
    
    def _inverse(self, y):
        r = tf.abs(y - self.x0)
        zhat = (y - self.x0) / r
        return self.b * r * zhat * h(self.a, r)
    
    def _forward_log_det_jacobian(self, y):
        try:
            n_dims = y.shape[1]
        except IndexError as e:
            raise RuntimeError('Input is one dimensional!')
        r = tf.abs(y - self.x0)
        dh = h_prime(self.a, r)
        hh = h(self.a, r)
        return (1 + self.b * hh)**2 * (1 + self.b * hh + self.b * dh * r)
    
    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))