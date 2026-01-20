
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

class train(tf.keras.Model):
    def __init__(self, vecpar, shared_layers):
        super(train, self).__init__()
        # set the value of parameters from vecpar
        self.vecpar = vecpar
        self.net = NeuralNetwork(shared_layers)

    def call(self, Y2):

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

        sample = tf.shape(Y2)[0]

        Y2 = tf.cast(Y2, dtype=tf.float64)

        q1, q2, sig_q1, sig_q2 = self.net(Y2)

        for n in range(0, int(period)):

            dZ = tf.sqrt(dt)*tf.random.normal(shape=[sample, 1], mean=0, stddev=1, dtype=tf.float64)

            q1_hat = q1/( q1 + q2 )
            q2_hat = 1 - q1_hat

            c = B*q1 - C*q1**2 + Y2
            W = q1 + q2

            sig_c = B*q1*sig_q1 - 2*C*q1**2*sig_q1 + sigma
            sig_W = q1*sig_q1 + q2*sig_q2
            sig_J = gamma/(gamma - 1)*( sig_c/c - sig_W/W )

            premium1 = gamma*( q1_hat*sig_q1**2 + q2_hat*sig_q1*sig_q2 ) - \
                       (1 - gamma)*sig_q1*sig_J
            premium2 = gamma*( q1_hat*sig_q1*sig_q2 + q2_hat*sig_q2**2 ) - \
                       (1 - gamma)*sig_q2*sig_J

            mu_Y2 = kappa*(Y_bar - Y2)

            r = rho + gamma*( ( B*q1 - 2*C*q1**2 )*( premium1 - (B - C*q1) ) - C*q1**2*sig_q1**2 + mu_Y2 )/c -\
                gamma*(gamma + 1)/2*sig_c**2/c**2

            r = r/( 1 - gamma*( B*q1 - 2*C*q1**2 )/c )

            mu_q1 = r + premium1 - ( B - C*q1 )
            mu_q2 = r + premium2 - Y2/q2

            q1 = q1*( 1 + mu_q1*dt + sig_q1*dZ )
            q2 = q2*( 1 + mu_q2*dt + sig_q2*dZ )
            Y2 = Y2 + kappa*(Y_bar - Y2)*dt + sigma*dZ

            q1_tilde, q2_tilde, sig_q1, sig_q2 = self.net(Y2)

            loss += tf.reduce_sum( (q1 - q1_tilde)**2 + (q2 - q2_tilde)**2 )/tf.cast(sample*period, dtype=tf.float64)

        return loss


class train_solver(tf.Module):  # network training

    def __init__(self, vecpar, initial_state, shared_layers):

        self.vecpar = vecpar

        self.initial_state = tf.cast(initial_state, dtype=tf.float32)
        self.model = train(vecpar, shared_layers)
        self.model.net.load_weights('./ckpt/Tree_sig_q_init/my_checkpoint')

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

            if (it+1) % 20 == 0:

                train_loss = train_loss/(batch + 1)
                elapsed = time.time() - start_time

                print('iter {} loss {} time {} lr {}'.format(it + 1, train_loss, elapsed, learning_rate))

                if train_loss <= train_loss_threshold:
                    self.model.net.save_weights('./ckpt/Tree_sig_q_minloss/my_checkpoint')
                    train_loss_threshold = train_loss

                self.model.net.save_weights('./ckpt/Tree_sig_q/my_checkpoint')

                self.plot_and_save()

    def plot_and_save(self):

        Y = np.expand_dims(np.linspace(0.2, 4, 500), 1)
        Y = tf.cast(Y, dtype=tf.float64)


        q1, q2, sig_q1, sig_q2 = self.model.net(Y)

        q1_hat = q1/( q1 + q2 )
        q2_hat = 1 - q1_hat

        c = B*q1 - C*q1**2 + Y
        W = q1 + q2

        sig_c = B*q1*sig_q1 - 2*C*q1**2*sig_q1 + sigma
        sig_W = q1*sig_q1 + q2*sig_q2
        sig_J = gamma/(gamma - 1)*( sig_c/c -sig_W/W )

        premium1 = gamma*( q1_hat*sig_q1**2 + q2_hat*sig_q1*sig_q2 ) - \
                   (1 - gamma)*sig_q1*sig_J

        mu_Y2 = kappa*(Y_bar - Y)

        r = rho + gamma*( ( B*q1 - 2*C*q1**2 )*( premium1 - (B - C*q1) ) - C*q1**2*sig_q1**2 + mu_Y2 )/c -\
                  gamma*(gamma + 1)/2*sig_c**2/c**2

        r = r/( 1 - gamma*( B*q1 - 2*C*q1**2 )/c )

        Y = np.linspace(0.2, 4, 500)

        q1 = q1.numpy().squeeze()
        q2 = q2.numpy().squeeze()
        r = r.numpy().squeeze()
        sig_q1 = sig_q1.numpy().squeeze()
        sig_q2 = sig_q2.numpy().squeeze()
        c = c.numpy().squeeze()
        sig_c = sig_c.numpy().squeeze()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]

        color_q1 = 'blue'
        ax1.plot(Y, q1, color=color_q1, linewidth=2, label='q1')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('q1', color=color_q1)
        ax1.tick_params(axis='y', labelcolor=color_q1)
        ax1.grid(True, alpha=0.3)

        ax1_right = ax1.twinx()
        color_sig_q1 = 'red'
        sig_q1_times_q1 = sig_q1 * q1
        ax1_right.plot(Y, sig_q1_times_q1, color=color_sig_q1, linewidth=2, linestyle='--',
                       label=r'$\sigma^{q,1} \times q1$')
        ax1_right.set_ylabel(r'$\sigma^{q,1} \times q1$', color=color_sig_q1)
        ax1_right.tick_params(axis='y', labelcolor=color_sig_q1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.set_title('(a) q1 and its volatility')

        ax2 = axes[0, 1]

        color_q2 = 'blue'
        ax2.plot(Y, q2, color=color_q2, linewidth=2, label='q2')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('q2', color=color_q2)
        ax2.tick_params(axis='y', labelcolor=color_q2)
        ax2.grid(True, alpha=0.3)

        ax2_right = ax2.twinx()
        color_sig_q2 = 'red'
        sig_q2_times_q2 = sig_q2 * q2
        ax2_right.plot(Y, sig_q2_times_q2, color=color_sig_q2, linewidth=2, linestyle='--',
                       label=r'$\sigma^{q,2} \times q2$')
        ax2_right.set_ylabel(r'$\sigma^{q,2} \times q2$', color=color_sig_q2)
        ax2_right.tick_params(axis='y', labelcolor=color_sig_q2)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.set_title('(b) q2 and its volatility')


        ax3 = axes[1, 0]
        color_c = 'blue'
        ax3.plot(Y, c, color=color_c, linewidth=2, label='c')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('c', color=color_c)
        ax3.tick_params(axis='y', labelcolor=color_c)
        ax3.grid(True, alpha=0.3)


        ax3_right = ax3.twinx()
        color_sig_c = 'red'
        ax3_right.plot(Y, sig_c, color=color_sig_c, linewidth=2, linestyle='--',
                       label=r'$\sigma_c$')
        ax3_right.set_ylabel(r'$\sigma_c$', color=color_sig_c)
        ax3_right.tick_params(axis='y', labelcolor=color_sig_c)


        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_right.get_legend_handles_labels()
        ax3.set_title('(c) Consumption and its volatility')


        ax4 = axes[1, 1]
        color_r = 'darkred'
        ax4.plot(Y, r, color=color_r, linewidth=3, label='Interest Rate (r)')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('r', color=color_r)
        ax4.tick_params(axis='y', labelcolor=color_r)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('(d) Interest Rate')

        plt.tight_layout()
        plt.savefig('BSDE_sig_q.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        data_dict = {
            'Y': Y,
            'q1': q1,
            'q2': q2,
            'sigma_q1': sig_q1,
            'sigma_q2': sig_q2,
            'c': c,
            'sigma_c': sig_c,
            'r': r,
            'sigma_q1_times_q1': sig_q1*q1,
            'sigma_q2_times_q2': sig_q2*q2
        }

        df_results = pd.DataFrame(data_dict)
        df_results.to_csv('BSDE_sig_q.csv', index=False)

N_sample = 50000
batch_size = 128
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
shared_layers = [32]

Y = np.random.uniform(0.05, 5, size=[N_sample, 1])

model = train_solver(vecpar, Y, shared_layers)

model.train(epoch=200, learning_rate=1e-3, batch_size=batch_size)
model.train(epoch=200, learning_rate=1e-4, batch_size=batch_size)
model.train(epoch=500, learning_rate=1e-5, batch_size=batch_size)
model.train(epoch=500, learning_rate=1e-6, batch_size=batch_size)