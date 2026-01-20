import os
import numpy as np
import tensorflow as tf
import time
import keras.layers
import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense
from math import isnan
tf.keras.backend.set_floatx('float64')

class NeuralNetwork(tf.keras.Model):
    def __init__(self, shared_layers):

        super(NeuralNetwork, self).__init__()
        # use GlorotNormal to initialize network parameters randomly
        initializer = tf.keras.initializers.GlorotNormal()

        self.shared_layers = [tf.keras.layers.Dense(shared_layers[i],
                                                    kernel_initializer=initializer,
                                                    bias_initializer='zeros',
                                                    activation=None)
                              for i in range(len(shared_layers))]

        self.q1_layers = tf.keras.layers.Dense(32,
                                                   kernel_initializer=initializer,
                                                   bias_initializer='zeros',
                                                   activation=None)

        self.q2_layers = tf.keras.layers.Dense(32,
                                                   kernel_initializer=initializer,
                                                   bias_initializer='zeros',
                                                   activation=None)

        self.sig_q1_layers = tf.keras.layers.Dense(32,
                                          kernel_initializer=initializer,
                                          bias_initializer='zeros',
                                          activation=None)

        self.sig_q2_layers = tf.keras.layers.Dense(32,
                                          kernel_initializer=initializer,
                                          bias_initializer='zeros',
                                          activation=None)

        self.q1_net = tf.keras.layers.Dense(1, kernel_initializer=initializer,bias_initializer='zeros', activation=None)
        self.q2_net = tf.keras.layers.Dense(1, kernel_initializer=initializer,bias_initializer='zeros', activation=None)

        self.sig_q1_net = tf.keras.layers.Dense(1, kernel_initializer=initializer,bias_initializer='zeros', activation=None)
        self.sig_q2_net = tf.keras.layers.Dense(1, kernel_initializer=initializer,bias_initializer='zeros', activation=None)

    def call(self, input):

        x = input

        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            x = tf.nn.tanh(x)

        q1 = self.q1_layers(x)
        q1 = tf.nn.tanh(q1)
        q1 = tf.exp(self.q1_net(q1))

        q2 = self.q2_layers(x)
        q2 = tf.nn.tanh(q2)
        q2 = tf.exp(self.q2_net(q2))

        sig_q1 = self.sig_q1_layers(x)
        sig_q1 = tf.nn.tanh(sig_q1)
        sig_q1 = self.sig_q1_net(sig_q1)

        sig_q2 = self.sig_q2_layers(x)
        sig_q2 = tf.nn.tanh(sig_q2)
        sig_q2 = self.sig_q2_net(sig_q2)

        return q1, q2, sig_q1, sig_q2


B = 1
C = 0.1

N_init = 100000
Y_init = np.random.uniform(0.1, 10, size=[N_init, 1])

class initialize_network(tf.keras.Model):

    def __init__(self, shared_layers):
        super(initialize_network, self).__init__()
        self.net = NeuralNetwork(shared_layers)

    def call(self, Y):

        sample = len(Y)
        Y = tf.cast(Y, dtype=tf.float64)

        q1, q2, sig_q1, sig_q2 = self.net(Y)

        # just guess q1 = 5 and q2 = 50 at any state
        loss = tf.reduce_sum( ( q1 - B/2/C )**2 )/sample +\
               tf.reduce_sum( ( q2 - 50 )**2 )/sample + \
               tf.reduce_sum( sig_q1**2 + sig_q2**2 )/sample

        return loss

class initialize_network_solver(tf.keras.Model):

    def __init__(self, shared_layers):
        super(initialize_network_solver, self).__init__()
        self.model = initialize_network(shared_layers)

    @tf.function
    def grad(self, Y_batch):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(Y_batch)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return grad, loss

    def train(self, Y_init, epoch, learning_rate, batch_size):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
        start_time = time.time()
        print('Training history(initialize network): ')
        dataset = tf.data.Dataset.from_tensor_slices((Y_init))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        for it in range(epoch):
            train_loss = 0
            for (batch, (Y_init_batch)) in enumerate(dataset):
                grad, loss = self.grad(Y_init_batch)
                optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
                train_loss += loss.numpy()

            if (it + 1) % 1 == 0:  # Log for each iteration

                elapsed_time = time.time() - start_time
                train_loss = train_loss/(batch + 1)
                print('Iter: %d, Loss: %.3e, Time: %.2f' % (it + 1, train_loss, elapsed_time))
                start_time = time.time()
                self.model.net.save_weights('./ckpt/Tree_sig_q_init/my_checkpoint')

shared_layers = [32]
initialize = initialize_network_solver(shared_layers)
initialize.train(Y_init, epoch=5, learning_rate=1e-3, batch_size=512)


