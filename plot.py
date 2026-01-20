import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载三个CSV文件
file_names = ['BSDE_AD', 'BSDE_sig_q', 'Iterative Method']
dataframes = {}

for file in file_names:
    # 读取CSV文件
    df = pd.read_csv(f'{file}.csv')
    dataframes[file] = df

# 检查每个DataFrame的列
for name, df in dataframes.items():
    print(f"{name} 的列: {list(df.columns)}")
    print(f"{name} 的形状: {df.shape}")
    print(f"前3行数据:\n{df.head(3)}\n")

# 2. 定义要绘制的变量列表（排除Y）
variables = ['q1', 'q2', 'sigma_q1', 'sigma_q2', 'c', 'sigma_c',
             'r', 'sigma_q1_times_q1', 'sigma_q2_times_q2']

# 3. 创建美观的变量名映射（用于标题和标签）
var_name_map = {
    'q1': r'$q_1$',
    'q2': r'$q_2$',
    'sigma_q1': r'$\sigma^{q,1}$',
    'sigma_q2': r'$\sigma^{q,2}$',
    'c': r'$c$',
    'sigma_c': r'$\sigma_c$',
    'r': r'$r$',
    'sigma_q1_times_q1': r'$\sigma^{q,1} \times q_1$',
    'sigma_q2_times_q2': r'$\sigma^{q,2} \times q_2$'
}

# 创建子图
n_vars = len(variables)
n_cols = 3  # 每行3个子图
n_rows = (n_vars + n_cols - 1) // n_cols  # 计算行数

# 创建图形
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
fig.suptitle('Comparison of Variables Across Three Methods', fontsize=16, y=1.02)

# 如果只有一行，axes是一维数组
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# 定义颜色和线型
colors = {'BSDE_AD': 'blue', 'BSDE_sig_q': 'red', 'Iterative Method': 'green'}
linestyles = {'BSDE_AD': '-', 'BSDE_sig_q': '--', 'Iterative Method': ':'}
labels = {'BSDE_AD': 'BSDE_AD', 'BSDE_sig_q': 'BSDE_sig_q', 'Iterative Method': 'Iterative Method'}

# 4. 绘制每个变量
for idx, var in enumerate(variables):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    # 用于存储数据点，用于自动设置y轴范围
    # 注意：这里排除'Iterative Method'的数据
    y_values_for_range = []

    # 检查每个文件中是否存在该变量
    for method_name, df in dataframes.items():
        if var in df.columns and 'Y' in df.columns:
            # 获取数据
            x_data = df['Y']
            y_data = df[var]

            # 如果是BSDE_AD或BSDE_sig_q，添加到y轴范围计算
            if method_name in ['BSDE_AD', 'BSDE_sig_q']:
                y_values_for_range.extend(y_data.values)

            # 使用Y列作为横坐标
            ax.plot(x_data, y_data, label=labels[method_name],
                    color=colors[method_name],
                    linestyle=linestyles[method_name],
                    linewidth=2)
        elif var not in df.columns:
            print(f"警告: {method_name} 中没有变量 {var}")
        elif 'Y' not in df.columns:
            print(f"警告: {method_name} 中没有Y列")

    # 自动设置Y轴范围（基于BSDE_AD和BSDE_sig_q的数据）
    if y_values_for_range:
        y_min = np.min(y_values_for_range)
        y_max = np.max(y_values_for_range)
        y_range = y_max - y_min

        # 如果y_range为0（所有值相同），添加小的偏移
        if y_range == 0:
            y_min = y_min - 0.1
            y_max = y_max + 0.1
        else:
            # 添加一些边距（5%）
            margin = y_range * 0.05
            y_min = y_min - margin
            y_max = y_max + margin

        ax.set_ylim([y_min, y_max])

    # 设置标题和标签（使用美观的变量名）
    display_name = var_name_map.get(var, var)
    ax.set_title(display_name, fontsize=12, fontweight='bold')
    if row ==2:
        ax.set_xlabel('Y', fontsize=10)
    ax.grid(True, alpha=0.3)
    if idx == 0:
       ax.legend(fontsize=9)

    # 设置x轴范围
    ax.set_xlim([0.2, 4])

    # 设置合适的刻度
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

# 5. 如果子图数量不是variables的整数倍，隐藏多余的子图
for idx in range(len(variables), n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].axis('off')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('Tree.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
