import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_eager_execution() 
import os 

import matplotlib.pyplot as plt 
plt.style.use('seaborn')

tfd = tfp.distributions
tfb = tfp.bijectors

class Planar(tfb.Bijector, tf.Module):

    def __init__(self, input_dimensions, case='density_estimation', validate_args=False, name='planar_flow'):
        """ usage of bijector inheritance """
        super(Planar, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            validate_args=validate_args,
            name=name)

        self.event_ndims = 1
        self.case = case

        try:
            assert self.case != 'density_estimation' or self.case != 'sampling'
        except ValueError:
            print('Case is not defined. Available options for case: density_estimation, sampling')

        self.u = tf.Variable(np.random.uniform(-1., 1., size=(int(input_dimensions))), name='u', dtype=tf.float32, trainable=True)
        self.w = tf.Variable(np.random.uniform(-1., 1., size=(int(input_dimensions))), name='w', dtype=tf.float32, trainable=True)
        self.b = tf.Variable(np.random.uniform(-1., 1., size=(1)), name='b', dtype=tf.float32, trainable=True)


    def h(self, y):
        return tf.math.tanh(y)

    def h_prime(self, y):
        return 1.0 - tf.math.tanh(y) ** 2.0

    def alpha(self):
        wu = tf.tensordot(self.w, self.u, 1)
        m = -1.0 + tf.nn.softplus(wu)
        return m - wu

    def _u(self):
        if tf.tensordot(self.w, self.u, 1) <= -1:
            alpha = self.alpha()
            z_para = tf.transpose(alpha * self.w / tf.math.sqrt(tf.reduce_sum(self.w ** 2.0)))
            self.u.assign_add(z_para)  # self.u = self.u + z_para

    def _forward_func(self, zk):
        inter_1 = self.h(tf.tensordot(zk, self.w, 1) + self.b)
        return tf.add(zk, tf.tensordot(inter_1, self.u, 0))

    def _forward(self, zk):
        if self.case == 'sampling':
            return self._forward_func(zk)
        else:
            raise NotImplementedError('_forward is not implemented for density_estimation')

    def _inverse(self, zk):
        if self.case == 'density_estimation':
            return self._forward_func(zk)
        else:
            raise NotImplementedError('_inverse is not implemented for sampling')
            
    def _log_det_jacobian(self, zk):
        psi = tf.tensordot(self.h_prime(tf.tensordot(zk, self.w, 1) + self.b), self.w, 0)
        det = tf.math.abs(1.0 + tf.tensordot(psi, self.u, 1))
        return tf.math.log(det)

    def _forward_log_det_jacobian(self, zk):
        if self.case == 'sampling':
            return -self._log_det_jacobian(zk)
        else:
            raise NotImplementedError('_forward_log_det_jacobian is not implemented for density_estimation')

    def _inverse_log_det_jacobian(self, zk):
        return self._log_det_jacobian(zk)