import numpy as np 
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
import os 

class Experiment:
    
    def __init__(self, optimizer, learning_rate, loss, steps=int(1e5)):
        self.optimizer = optimizer
        self.steps = int(steps)
        
    def change_optimizer(self, learning_rate, loss, keyword='adam'):
        if keyword == 'adam':
            self.optimizer = self.optimizer
        elif keyword == 'sgd':
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif keyword == 'rmsprop':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss)
        else:
            raise NotImplementedError('Undefined optimizer!')
            
    def get_optimizer(self):
        return self.optimizer
    
    def set_iteration_count(self, iteration_count):
        self.steps = iteration_count
        
    def get_iteration_count(self):
        return self.steps