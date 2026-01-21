
import os
import numpy as np
import tensorflow as tf
import time
import keras.layers
import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense
from math import isnan
import matplotlib.pyplot as plt
import pandas as pd
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

        self.q3_layers = tf.keras.layers.Dense(32,
                                                   kernel_initializer=initializer,
                                                   bias_initializer='zeros',
                                                   activation=None)

        self.q1_net = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer='zeros', activation=None)
        self.q2_net = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer='zeros', activation=None)
        self.q3_net = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer='zeros', activation=None)

    def call(self, Y2, Y3):

        x = tf.concat([Y2, Y3], axis=1)

        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            x = tf.nn.tanh(x)

        q1 = self.q1_layers(x)
        q1 = tf.nn.tanh(q1)
        q1 = tf.exp(self.q1_net(q1))

        q2 = self.q2_layers(x)
        q2 = tf.nn.tanh(q2)
        q2 = tf.exp(self.q2_net(q2))

        q3 = self.q3_layers(x)
        q3 = tf.nn.tanh(q3)
        q3 = tf.exp(self.q3_net(q3))

        return q1, q2, q3

class train(tf.keras.Model):
    def __init__(self, vecpar, shared_layers):
        super(train, self).__init__()
        # set the value of parameters from vecpar
        self.vecpar = vecpar
        self.net = NeuralNetwork(shared_layers)

    def call(self, Y):

        loss = 0.0
        #                      0        1      2    3  4    5     6     7    8
        #vecpar = np.array([N_period, kappa, Y_bar, B, C, gamma, rho, sigma, dt])

        period = self.vecpar[0]
        kappa = tf.cast(self.vecpar[1], dtype=tf.float64)
        Y_bar = tf.cast(self.vecpar[2], dtype=tf.float64)
        B = tf.cast(self.vecpar[3], dtype=tf.float64)
        C = tf.cast(self.vecpar[4], dtype=tf.float64)
        gamma = tf.cast(self.vecpar[5], dtype=tf.float64)
        rho = tf.cast(self.vecpar[6], dtype=tf.float64)
        sigma = tf.cast(self.vecpar[7], dtype=tf.float64)
        dt = tf.cast(self.vecpar[8], dtype=tf.float64)

        sample = tf.shape(Y)[0]

        Y = tf.cast(Y, dtype=tf.float64)

        Y2 = tf.expand_dims(Y[:, 0], 1)
        Y3 = tf.expand_dims(Y[:, 1], 1)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([Y2, Y3])  # 同时监视两个变量
            q1, q2, q3 = self.net(Y2, Y3)

        dq1_dY2 = tape.gradient(q1, Y2)
        dq2_dY2 = tape.gradient(q2, Y2)
        dq3_dY2 = tape.gradient(q3, Y2)

        dq1_dY3 = tape.gradient(q1, Y3)
        dq2_dY3 = tape.gradient(q2, Y3)
        dq3_dY3 = tape.gradient(q3, Y3)

        del tape

        sig_q1_Z1 = dq1_dY2*sigma/q1
        sig_q2_Z1 = dq2_dY2*sigma/q2
        sig_q3_Z1 = dq3_dY2*sigma/q3

        sig_q1_Z2 = dq1_dY3*sigma/q1
        sig_q2_Z2 = dq2_dY3*sigma/q2
        sig_q3_Z2 = dq3_dY3*sigma/q3

        for n in range(0, int(period)):

            dZ1 = tf.sqrt(dt)*tf.random.normal(shape=[sample, 1], mean=0, stddev=1, dtype=tf.float64)
            dZ2 = tf.sqrt(dt)*tf.random.normal(shape=[sample, 1], mean=0, stddev=1, dtype=tf.float64)

            q1_hat = q1/( q1 + q2 + q3 )
            q2_hat = q2/( q1 + q2 + q3 )
            q3_hat = 1 - q1_hat - q2_hat

            c = B*q1 - C*q1**2 + Y2 + Y3
            W = q1 + q2 + q3

            sig_c_Z1 = B*q1*sig_q1_Z1 - 2*C*q1**2*sig_q1_Z1 + sigma
            sig_c_Z2 = B*q1*sig_q1_Z2 - 2*C*q1**2*sig_q1_Z2 + sigma

            sig_W_Z1 = q1*sig_q1_Z1 + q2*sig_q2_Z1 + q3*sig_q3_Z1
            sig_W_Z2 = q1*sig_q1_Z2 + q2*sig_q2_Z2 + q3*sig_q3_Z2

            sig_J_Z1 = gamma/(gamma - 1)*( sig_c_Z1/c - sig_W_Z1/W )
            sig_J_Z2 = gamma/(gamma - 1)*( sig_c_Z2/c - sig_W_Z2/W )

            premium1 = gamma*q1_hat*( sig_q1_Z1**2 + sig_q1_Z2**2 ) +\
                       gamma*q2_hat*( sig_q1_Z1*sig_q2_Z1 + sig_q1_Z2*sig_q2_Z2 ) +\
                       gamma*q3_hat*( sig_q1_Z1*sig_q3_Z1 + sig_q1_Z2*sig_q3_Z2 ) - \
                       (1 - gamma)*( sig_J_Z1*sig_q1_Z1 + sig_J_Z2*sig_q1_Z2 )

            premium2 = gamma*q1_hat*( sig_q1_Z1*sig_q2_Z1 + sig_q1_Z2*sig_q2_Z2 ) +\
                       gamma*q2_hat*( sig_q2_Z1**2 + sig_q2_Z2**2 ) +\
                       gamma*q3_hat*( sig_q2_Z1*sig_q3_Z1 + sig_q2_Z2*sig_q3_Z2 ) - \
                       (1 - gamma)*( sig_J_Z1*sig_q2_Z1 + sig_J_Z2*sig_q2_Z2 )

            premium3 = gamma*q1_hat*( sig_q1_Z1*sig_q3_Z1 + sig_q1_Z2*sig_q3_Z2 ) +\
                       gamma*q2_hat*( sig_q2_Z1*sig_q3_Z1 + sig_q2_Z2*sig_q3_Z2 ) +\
                       gamma*q3_hat*( sig_q3_Z1**2 + sig_q3_Z2**2 ) - \
                       (1 - gamma)*( sig_J_Z1*sig_q3_Z1 + sig_J_Z2*sig_q3_Z2 )

            mu_Y2 = kappa*(Y_bar - Y2)
            mu_Y3 = kappa*(Y_bar - Y3)

            r = rho + gamma*( ( B*q1 - 2*C*q1**2 )*( premium1 - (B - C*q1) ) - C*q1**2*( sig_q1_Z1**2 + sig_q1_Z2**2 ) + mu_Y2 + mu_Y3 )/c -\
                      gamma*(gamma + 1)/2*( sig_c_Z1**2 + sig_c_Z2**2 )/c**2

            r = r/( 1 - gamma*( B*q1 - 2*C*q1**2 )/c )

            mu_q1 = r + premium1 - ( B - C*q1 )
            mu_q2 = r + premium2 - Y2/q2
            mu_q3 = r + premium3 - Y3/q3

            q1 = q1*( 1 + mu_q1*dt + sig_q1_Z1*dZ1 + sig_q1_Z2*dZ2 )
            q2 = q2*( 1 + mu_q2*dt + sig_q2_Z1*dZ1 + sig_q2_Z2*dZ2 )
            q3 = q3*( 1 + mu_q3*dt + sig_q3_Z1*dZ1 + sig_q3_Z2*dZ2 )

            Y2 = Y2 + kappa*(Y_bar - Y2)*dt + sigma*dZ1
            Y3 = Y3 + kappa*(Y_bar - Y3)*dt + sigma*dZ2

            with tf.GradientTape(persistent=True) as tape:
                tape.watch([Y2, Y3])  # 同时监视两个变量
                q1_tilde, q2_tilde, q3_tilde = self.net(Y2, Y3)

            dq1_dY2 = tape.gradient(q1_tilde, Y2)
            dq2_dY2 = tape.gradient(q2_tilde, Y2)
            dq3_dY2 = tape.gradient(q3_tilde, Y2)

            dq1_dY3 = tape.gradient(q1_tilde, Y3)
            dq2_dY3 = tape.gradient(q2_tilde, Y3)
            dq3_dY3 = tape.gradient(q3_tilde, Y3)

            del tape

            sig_q1_Z1 = dq1_dY2*sigma/q1_tilde
            sig_q2_Z1 = dq2_dY2*sigma/q2_tilde
            sig_q3_Z1 = dq3_dY2*sigma/q3_tilde

            sig_q1_Z2 = dq1_dY3*sigma/q1_tilde
            sig_q2_Z2 = dq2_dY3*sigma/q2_tilde
            sig_q3_Z2 = dq3_dY3*sigma/q3_tilde

            loss += tf.reduce_sum( (q1 - q1_tilde)**2 + (q2 - q2_tilde)**2 + (q3 - q3_tilde)**2 )/tf.cast(sample*period, dtype=tf.float64)

        return loss


class train_solver(tf.Module):  # network training

    def __init__(self, vecpar, initial_state, shared_layers):

        self.vecpar = vecpar

        self.initial_state = tf.cast(initial_state, dtype=tf.float32)
        self.model = train(vecpar, shared_layers)
        self.model.net.load_weights('./ckpt/Tree2_AD_init/my_checkpoint')

    @tf.function
    def grad(self, initial_state_batch):

        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(initial_state_batch)  # calculate loss function
        grad = tape.gradient(loss, self.model.net.trainable_variables)  # calculate gradients of loss to network parameters
        del tape

        return grad, loss

    def train(self, epoch, batch_size, learning_rate):
        train_loss_threshold = 10
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)

        for it in range(epoch):

            start_time = time.time()

            dataset = tf.data.Dataset.from_tensor_slices(self.initial_state)
            dataset = dataset.shuffle(N_sample, reshuffle_each_iteration=True)
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
            train_loss = 0

            for (batch, (initial_state_batch)) in enumerate(dataset):
                  flag_nan = False
                  grad, loss = self.grad(initial_state_batch)

                  for g in grad:
                      if np.isnan(g.numpy()).any() or np.isinf(g.numpy()).any():
                          flag_nan = True

                  if flag_nan == False:
                      optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.net.trainable_variables))

                  train_loss += loss.numpy()

            if (it+1) % 10 == 0:

                train_loss = train_loss/(batch + 1)
                elapsed = time.time() - start_time

                print('iter {} loss {} time {} lr {}'.format(it + 1, train_loss, elapsed, learning_rate))

                if train_loss <= train_loss_threshold:
                    self.model.net.save_weights('./ckpt/Tree2_AD_minloss/my_checkpoint')
                    train_loss_threshold = train_loss



# the total sample size is 102400
N_sample = 50000
batch_size = 128
# dt is the length of each period
dt = 0.005
N_period = 30
kappa = 0.2
Y_bar = 2
gamma = 1.2
rho = 0.03
sigma = 0.03
B = 1
C = 0.1

vecpar = np.array([N_period, kappa, Y_bar, B, C, gamma, rho, sigma, dt])
shared_layers = [64]

Y = np.random.uniform(0.05, 5, size=[N_sample, 2])

model = train_solver(vecpar, Y, shared_layers)

model.train(epoch=300, learning_rate=1e-3, batch_size=batch_size)
model.train(epoch=300, learning_rate=1e-4, batch_size=batch_size)
model.train(epoch=500, learning_rate=1e-5, batch_size=batch_size)