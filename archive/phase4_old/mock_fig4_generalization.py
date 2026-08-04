import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# 确保输出目录存在
output_dir = r'd:\NEOschool\2026BMI-\figures'
os.makedirs(output_dir, exist_ok=True)

# 设置学术图表样式 (符合顶会规范)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

# X轴数据: 扰动比例 (0.0 到 1.0)
x = np.linspace(0, 1, 11)
x_smooth = np.linspace(0, 1, 300)

def smooth_curve(x, y):
    """三次样条平滑插值"""
    spl = make_interp_spline(x, y, k=3)
    return np.clip(spl(x_smooth), 0, 1.1)

# ================= 模拟数据生成 =================
# Spectral (频域): 中等程度依赖，平缓指数衰减
y_spec = np.exp(-1.5 * x)
std_spec = 0.05 + 0.1 * x # 个体组间的方差逐渐增大

# Spatial (空域): 高度依赖，Sigmoid型快速崩溃
y_spat = 1 / (1 + np.exp(12 * (x - 0.25)))
std_spat = 0.08 + 0.15 * np.sin(x * np.pi)

# Temporal (时域): 低依赖，具备高度鲁棒性，线性缓慢下降
y_temp = 1 - 0.15 * x
std_temp = 0.04 + 0.06 * x

# ================= 绘图逻辑 =================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
titles = ['(A) Spectral Dependence', '(B) Spatial Dependence', '(C) Temporal Dependence']
y_data = [y_spec, y_spat, y_temp]
std_data = [std_spec, std_spat, std_temp]
colors = ['#1f77b4', '#d62728', '#2ca02c'] # 蓝, 红, 绿

# 绘制 A, B, C 三个维度的子图
for i, ax in enumerate(axes.flatten()[:3]):
    # 平滑拟合线 (均值)
    ax.plot(x_smooth, smooth_curve(x, y_data[i]), color=colors[i], linewidth=2.5, label='Mean Retention')
    
    # 个体组方差的阴影带 (95% CI)
    ax.fill_between(x_smooth, 
                    smooth_curve(x, y_data[i] - std_data[i]), 
                    smooth_curve(x, y_data[i] + std_data[i]), 
                    color=colors[i], alpha=0.15, label='95% CI (Across Subjects)')
    
    # 原始散点
    ax.scatter(x, y_data[i], color=colors[i], s=60, zorder=5, edgecolor='w', linewidth=1.5)
    
    # 图表装饰
    ax.set_title(titles[i], fontweight='bold', pad=15)
    ax.set_xlabel('Masking / Perturbation Ratio')
    ax.set_ylabel('Relative Performance Retention')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, edgecolor='k')

# 绘制 D: 跨维度综合对比图 (重叠呈现)
ax4 = axes.flatten()[3]
labels = ['Spectral (AUC=0.45)', 'Spatial (AUC=0.22)', 'Temporal (AUC=0.88)']
for i in range(3):
    ax4.plot(x_smooth, smooth_curve(x, y_data[i]), color=colors[i], linewidth=3, label=labels[i])

ax4.set_title('(D) Cross-Dimensional Comparison', fontweight='bold', pad=15)
ax4.set_xlabel('Masking / Perturbation Ratio')
ax4.set_ylabel('Relative Performance Retention')
ax4.set_ylim(0, 1.1)
ax4.set_xlim(0, 1.0)
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(loc='lower left', frameon=True, edgecolor='k', title='Dimension (Robustness)')

plt.tight_layout(pad=3.0)
save_path = os.path.join(output_dir, 'Fig4_Generalization_Mock.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Mock figure generated successfully at: {save_path}")
