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
from matplotlib.colors import LinearSegmentedColormap
tf.keras.backend.set_floatx('float64')

# 创建低蓝高红的自定义色彩映射
def create_blue_to_red_cmap():
    colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # 蓝 -> 青 -> 黄 -> 红
    positions = [0.0, 0.3, 0.7, 1.0]
    return LinearSegmentedColormap.from_list('blue_to_red', list(zip(positions, colors)))

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

shared_layers = [64]
net = NeuralNetwork(shared_layers)
net.load_weights('./ckpt/Tree2_AD_minloss/my_checkpoint')

kappa = 0.2
Y_bar = 2
gamma = 1.2
rho = 0.03
sigma = 0.03
B = 1
C = 0.1


# 创建测试数据网格
y2_range = np.linspace(0.2, 4, 100)  # 将[0.05, 4]分成100个点
y3_range = np.linspace(0.2, 4, 100)  # 将[0.05, 4]分成100个点
y2_grid, y3_grid = np.meshgrid(y2_range, y3_range)

# 将网格点展平以便批量预测
Y2 = tf.cast(y2_grid.flatten().reshape(-1, 1), dtype=tf.float64)
Y3 = tf.cast(y3_grid.flatten().reshape(-1, 1), dtype=tf.float64)


with tf.GradientTape(persistent=True) as tape:
    tape.watch([Y2, Y3])  # 同时监视两个变量
    q1, q2, q3 = net(Y2, Y3)

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

# 将结果转换回网格形状
q1_grid = q1.numpy().reshape(100, 100)
q2_grid = q2.numpy().reshape(100, 100)
q3_grid = q3.numpy().reshape(100, 100)
r_grid = r.numpy().reshape(100, 100)

# 创建自定义色彩映射
blue_to_red = create_blue_to_red_cmap()

# 创建绘图 - 4个子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 绘制q1的热力图
q1_min, q1_max = np.min(q1_grid), np.max(q1_grid)
im1 = axes[0, 0].imshow(q1_grid,
                       extent=[0.2, 4.0, 0.2, 4.0],
                       origin='lower',
                       aspect='auto',
                       cmap=blue_to_red,
                       vmin=q1_min, vmax=q1_max)  # 固定色彩范围
axes[0, 0].set_xlabel('Y2', fontsize=12)
axes[0, 0].set_ylabel('Y3', fontsize=12)
axes[0, 0].set_title('q1 ', fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# 绘制q2的热力图
q2_min, q2_max = np.min(q2_grid), np.max(q2_grid)
im2 = axes[0, 1].imshow(q2_grid,
                       extent=[0.2, 4.0, 0.2, 4.0],
                       origin='lower',
                       aspect='auto',
                       cmap=blue_to_red,
                       vmin=q2_min, vmax=q2_max)  # 固定色彩范围
axes[0, 1].set_xlabel('Y2', fontsize=12)
axes[0, 1].set_ylabel('Y3', fontsize=12)
axes[0, 1].set_title('q2 ', fontsize=14, fontweight='bold')
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# 绘制q3的热力图
q3_min, q3_max = np.min(q3_grid), np.max(q3_grid)
im3 = axes[1, 0].imshow(q3_grid,
                       extent=[0.2, 4.0, 0.2, 4.0],
                       origin='lower',
                       aspect='auto',
                       cmap=blue_to_red,
                       vmin=q3_min, vmax=q3_max)  # 固定色彩范围
axes[1, 0].set_xlabel('Y2', fontsize=12)
axes[1, 0].set_ylabel('Y3', fontsize=12)
axes[1, 0].set_title('q3 ', fontsize=14, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')


r_min, r_max = np.min(r_grid), np.max(r_grid)
im4 = axes[1, 1].imshow(r_grid,
                       extent=[0.2, 4.0, 0.2, 4.0],
                       origin='lower',
                       aspect='auto',
                       cmap=blue_to_red,
                       vmin=r_min, vmax=r_max)
axes[1, 1].set_xlabel('Y2', fontsize=12)
axes[1, 1].set_ylabel('Y3', fontsize=12)
axes[1, 1].set_title('risk free rate r', fontsize=14, fontweight='bold')
plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('BSDE_3Tree.png', dpi=300, bbox_inches='tight')
plt.show()