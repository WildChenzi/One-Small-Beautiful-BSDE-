import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

dt = 0.001
gamma = 1.2
rho = 0.03
sigma = 0.03
kappa = 0.2
B = 1
C = 0.1
Y_bar = 2

# use 3000 points distirbuted on [0, 10] which is state space for Y
N = 3000
T = 1000000

Y = np.linspace(0, 10, N)
dY = Y[1] - Y[0] #length of each interval

mu_Y = kappa*( Y_bar - Y )
sig_Y = sigma

df_q1 = np.zeros([N])
df_q2 = np.zeros([N])
dff_q1 = np.zeros([N])
dff_q2 = np.zeros([N])

q1 = 5 + 0.01*Y #terminal condition for q1
q2 = 0.5 + 0.01*Y #terminal condition for q2

lambda_val = 0.0005

def payoff_policy_growth(X, R, MU, S, G, V, lambda_val, bc_right):
    """
    隐式有限差分求解HJB方程

    参数:
    X: 状态空间网格 (递增序列), shape (N,)
    R: 贴现率减去增长率, shape (N,)
    MU: 漂移项, shape (N,),
    S: 波动率, shape (N,),
    G: 即时收益流, shape (N,)
    V: 未来价值函数 F(t+dt, X), shape (N,)
    lambda_val: 时间权重参数, lambda = dt/(1+dt)

    返回:
    F: 当前价值函数 F(t, X), shape (N,)
    """

    N = len(X)
    dX = X[1] - X[0]  # calculate again in function

    S0 = np.ones(N)*sigma
    S0 = S0**2/2/dX

    DU = np.zeros([N])  # upper diagonal
    DD = np.zeros([N - 1])

    DU[1:] = -( np.maximum(MU[:-1], 0) + S0[:-1] )/dX*lambda_val

    DD = -( np.maximum(-MU[1:], 0) + S0[1:] )/dX*lambda_val

    D0 = (1 - lambda_val) + lambda_val*R

    D0[:-1] = D0[:-1] - DU[1:]
    D0[1:] = D0[1:] - DD

    A = sparse.diags([D0, DU[1:], DD],
                     offsets=[0, 1, -1],
                     shape=(N, N),
                     format='csr')

    b = G*lambda_val + V*(1 - lambda_val)
    b[-1] = 0

    if bc_right == True: #Second-order derivative constraint

        A[-1, -3] = 1
        A[-1, -2] = -2
        A[-1, -1] = 1

    else: #First-order derivative constraint

        A[-1, -2] = -1
        A[-1, -1] = 1

    F = spsolve(A, b)

    return F

for i in range(T):

    df_q1[1:] = ( q1[1:] - q1[:-1] )/dY
    df_q2[1:] = ( q2[1:] - q2[:-1] )/dY

    df_q1[0] = ( q1[1] - q1[0] )/dY
    df_q2[0] = ( q2[1] - q2[0] )/dY

    dff_q1[1:] = ( df_q1[1:] - df_q1[:-1] )/dY
    dff_q2[1:] = ( df_q2[1:] - df_q2[:-1] )/dY
    dff_q1[0] = ( df_q1[1] - df_q1[0] )/dY
    dff_q2[0] = ( df_q2[1] - df_q2[0] )/dY

    c = B*q1 - C*q1**2 + Y
    W = q1 + q2
    x1 = q1/W
    x2 = q2/W

    sig_q1 = df_q1*sig_Y/q1
    sig_q2 = df_q2*sig_Y/q2

    sig_c = B*q1*sig_q1 - 2*C*q1**2*sig_q1 + sig_Y
    sig_W = q1*sig_q1 + q2*sig_q2

    sig_J = gamma/(gamma - 1)*( sig_c/c - sig_W/W )

    pi1 = gamma*( x1*sig_q1**2 + x2*sig_q1*sig_q2 ) - \
          (1 - gamma)*sig_q1*sig_J
    pi2 = gamma*( x1*sig_q1*sig_q2 + x2*sig_q2**2 ) - \
          (1 - gamma)*sig_q2*sig_J

    mu_q1 = pi1 + rho + gamma*( -C*q1**2*sig_q1**2 + mu_Y )/c - (B - C*q1) - gamma*(gamma + 1)/2*sig_c**2/c**2
    mu_q1 = mu_q1/( 1 - gamma*( B*q1 - 2*C*q1**2 )/c )
    r = rho + gamma*( ( B*q1 - 2*C*q1**2 )*mu_q1 - C*q1**2*sig_q1**2 + mu_Y )/c - gamma*(gamma + 1)/2*sig_c**2/c**2

    mu_q2 = r + pi2 - Y/q2

    q_old = np.stack((q1, q2), axis=1)

    q1 = payoff_policy_growth(Y, mu_q1, mu_Y, sigma, 0, q1, lambda_val, bc_right=False)
    q2 = payoff_policy_growth(Y, mu_q2, mu_Y, sigma, 0, q2, lambda_val, bc_right=True)

    q_new = np.stack((q1, q2), axis=1)

    if i%1000 == 0:
       eps = np.max(np.abs(q_new - q_old))
       print('eps = ', eps)

    if eps < 5e-5:
        break


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
sig_q1_times_q1 = sig_q1*q1
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
sig_q2_times_q2 = sig_q2*q2
ax2_right.plot(Y, sig_q2_times_q2, color=color_sig_q2, linewidth=2, linestyle='--',
               label=r'$\sigma^{q,2} \times q2$')
ax2_right.set_ylabel(r'$\sigma^{q,2} \times q2$', color=color_sig_q2)
ax2_right.tick_params(axis='y', labelcolor=color_sig_q2)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.set_title('(b) q2 and its volatility')

# 第三个子图：c左轴，sigma_c右轴
ax3 = axes[1, 0]
# 左轴：c
color_c = 'blue'
ax3.plot(Y, c, color=color_c, linewidth=2, label='c')
ax3.set_xlabel('Y')
ax3.set_ylabel('c', color=color_c)
ax3.tick_params(axis='y', labelcolor=color_c)
ax3.grid(True, alpha=0.3)

# 右轴：sigma_c
ax3_right = ax3.twinx()
color_sig_c = 'red'
ax3_right.plot(Y, sig_c, color=color_sig_c, linewidth=2, linestyle='--',
               label=r'$\sigma_c$')
ax3_right.set_ylabel(r'$\sigma_c$', color=color_sig_c)
ax3_right.tick_params(axis='y', labelcolor=color_sig_c)

# 合并图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.set_title('(c) Consumption and its volatility')

# 第四个子图：r
ax4 = axes[1, 1]
color_r = 'darkred'
ax4.plot(Y, r, color=color_r, linewidth=3, label='Interest Rate (r)')
ax4.set_xlabel('Y')
ax4.set_ylabel('r', color=color_r)
ax4.tick_params(axis='y', labelcolor=color_r)
ax4.grid(True, alpha=0.3)
ax4.set_title('(d) Interest Rate')


plt.tight_layout()
plt.savefig('Iterative Method.png', dpi=300, bbox_inches='tight')
plt.show()


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
df_results.to_csv('Iterative Method.csv', index=False)
