import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 确保输出目录存在
output_dir = r'd:\NEOschool\2026BMI-\figures'
os.makedirs(output_dir, exist_ok=True)

# 设置学术图表样式 (符合顶会规范)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

# ================= 模拟数据生成 =================
np.random.seed(42)
N_SUBJECTS = 30

# 1. 基线准确率 (Subject Accuracy)
# 假设随机分布在 0.55 到 0.95 之间
accuracies = np.random.uniform(0.55, 0.95, N_SUBJECTS)

# 2. 对 T2 x Beta 核心特征的依赖度 (Drop值)
# 设定其与准确率成较强的正相关 (准确率越高的被试，越依赖这个特征)
# Y = a*X + b + noise
t2_beta_drop = 0.8 * accuracies - 0.3 + np.random.normal(0, 0.05, N_SUBJECTS)
t2_beta_drop = np.clip(t2_beta_drop, 0.0, 0.6) # Drop值限制在合理范围内

# 3. 对次要/通用空间特征的依赖度 (Spatial Drop)
# 设定其与准确率成弱负相关或不相关 (低分被试可能更依赖通用拓扑)
spatial_drop = -0.3 * accuracies + 0.5 + np.random.normal(0, 0.08, N_SUBJECTS)
spatial_drop = np.clip(spatial_drop, 0.05, 0.5)

# 组合成 DataFrame
df = pd.DataFrame({
    'Subject': [f'S{i+1:02d}' for i in range(N_SUBJECTS)],
    'Accuracy': accuracies,
    'T2_Beta_Drop': t2_beta_drop,
    'Spatial_Drop': spatial_drop
})

# ================= 绘图逻辑 =================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = sns.color_palette("muted")

# ---------------------------------------------------------
# Panel A: Subject-level Performance Distribution (Violin + Swarm)
# ---------------------------------------------------------
ax = axes[0, 0]
sns.violinplot(y=df['Accuracy'], ax=ax, color='lightgray', inner=None, alpha=0.5)
sns.swarmplot(y=df['Accuracy'], ax=ax, color=colors[0], size=8, edgecolor='w', linewidth=1)
ax.set_title('(A) Subject-level Baseline Accuracy', fontweight='bold', pad=15)
ax.set_ylabel('Decoding Accuracy')
ax.set_xticks([])
ax.set_ylim(0.45, 1.0)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# ---------------------------------------------------------
# Panel B: Accuracy vs. T2xBeta Dependence
# ---------------------------------------------------------
ax = axes[0, 1]
sns.regplot(data=df, x='Accuracy', y='T2_Beta_Drop', ax=ax, 
            color=colors[3], scatter_kws={'s': 60, 'edgecolor': 'w'},
            line_kws={'linewidth': 2})
# 计算相关系数
r, p = pearsonr(df['Accuracy'], df['T2_Beta_Drop'])
ax.text(0.05, 0.85, f'r = {r:.2f}\np = {p:.1e}', transform=ax.transAxes, 
        fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
ax.set_title('(B) Accuracy vs. T2×Beta Dependence', fontweight='bold', pad=15)
ax.set_xlabel('Baseline Accuracy')
ax.set_ylabel('Accuracy Drop (T2×Beta Ablation)')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(-0.05, 0.65)
ax.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# Panel C: Accuracy vs. Spatial Dependence
# ---------------------------------------------------------
ax = axes[1, 0]
sns.regplot(data=df, x='Accuracy', y='Spatial_Drop', ax=ax, 
            color=colors[1], scatter_kws={'s': 60, 'edgecolor': 'w'},
            line_kws={'linewidth': 2, 'linestyle': '--'})
r_sp, p_sp = pearsonr(df['Accuracy'], df['Spatial_Drop'])
# p > 0.05 通常用 n.s. (not significant) 表示
p_text = f'p = {p_sp:.2f}' if p_sp < 0.001 else f'p = {p_sp:.3f}'
if p_sp >= 0.05: p_text += ' (n.s.)'

ax.text(0.65, 0.85, f'r = {r_sp:.2f}\n{p_text}', transform=ax.transAxes, 
        fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
ax.set_title('(C) Accuracy vs. Spatial Dependence', fontweight='bold', pad=15)
ax.set_xlabel('Baseline Accuracy')
ax.set_ylabel('Accuracy Drop (Spatial Ablation)')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.0, 0.6)
ax.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# Panel D: Feature-Feature Trade-off
# ---------------------------------------------------------
ax = axes[1, 1]
# 绘制 T2xBeta Drop vs Spatial Drop，用准确率映射颜色深浅
scatter = ax.scatter(df['T2_Beta_Drop'], df['Spatial_Drop'], 
                     c=df['Accuracy'], cmap='viridis', s=80, edgecolor='w', linewidth=1)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Baseline Accuracy')

# 添加拟合线
sns.regplot(data=df, x='T2_Beta_Drop', y='Spatial_Drop', ax=ax, scatter=False,
            color='gray', line_kws={'linewidth': 2, 'linestyle': ':'})

r_trade, p_trade = pearsonr(df['T2_Beta_Drop'], df['Spatial_Drop'])
ax.text(0.65, 0.85, f'r = {r_trade:.2f}', transform=ax.transAxes, 
        fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

ax.set_title('(D) Cross-Feature Strategy Trade-off', fontweight='bold', pad=15)
ax.set_xlabel('T2×Beta Dependence (Drop)')
ax.set_ylabel('Spatial Dependence (Drop)')
ax.set_xlim(-0.05, 0.65)
ax.set_ylim(0.0, 0.6)
ax.grid(True, linestyle='--', alpha=0.5)

# 调整布局并保存大图
plt.tight_layout(pad=3.0)
save_path = os.path.join(output_dir, 'Fig4_CrossSubject_Mock.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Mock figure generated successfully at: {save_path}")

# ================= 单独保存 Panel D =================
fig_d, ax_d = plt.subplots(figsize=(8, 6))
scatter_d = ax_d.scatter(df['T2_Beta_Drop'], df['Spatial_Drop'], 
                         c=df['Accuracy'], cmap='viridis', s=80, edgecolor='w', linewidth=1)
cbar_d = plt.colorbar(scatter_d, ax=ax_d)
cbar_d.set_label('Baseline Accuracy')

# 添加拟合线
sns.regplot(data=df, x='T2_Beta_Drop', y='Spatial_Drop', ax=ax_d, scatter=False,
            color='gray', line_kws={'linewidth': 2, 'linestyle': ':'})

ax_d.text(0.65, 0.85, f'r = {r_trade:.2f}', transform=ax_d.transAxes, 
          fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

ax_d.set_title('(D) Cross-Feature Strategy Trade-off', fontweight='bold', pad=15)
ax_d.set_xlabel('T2×Beta Dependence (Drop)')
ax_d.set_ylabel('Spatial Dependence (Drop)')
ax_d.set_xlim(-0.05, 0.65)
ax_d.set_ylim(0.0, 0.6)
ax_d.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
save_path_d = os.path.join(output_dir, 'Fig4_CrossSubject_Mock_PanelD.png')
plt.savefig(save_path_d, dpi=300, bbox_inches='tight')
print(f"Panel D generated successfully at: {save_path_d}")
