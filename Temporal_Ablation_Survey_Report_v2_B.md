# NeuroBridge Temporal 消融实验报告 v2-B
## EEG-to-Image 解码模型时频特征重要性分析：三档 Gamma 精细化方案

> **作者：** 乔钰成  
> **日期：** 2026年  
> **机构：** 华东师范大学 · NEOschool  
> **环境：** Python 3.10 · PyTorch 1.13 · NeuroBridge · THINGS-EEG2  
> **版本说明（v2-B 相对 v1 的主要变更）：**  
> 1. Gamma 频段从单一档细化为三档：low_gamma 30–45Hz / gamma 45–70Hz / high_gamma 70–100Hz  
> 2. 修正 hi_gamma 上限（原 80–120Hz → 70–100Hz，对齐数据集物理约束）  
> 3. 新增 Phase 3：单一时间窗内全频段 STFT 置零实验  

---

## 目录

1. [实验背景与目标](#1-实验背景与目标)
2. [频段划分修订说明](#2-频段划分修订说明)
3. [时间窗划分](#3-时间窗划分)
4. [实验设计总览（三阶段）](#4-实验设计总览)
5. [Phase 1：STFT 时频掩码（精细化 Gamma）](#5-phase-1)
6. [Phase 2：振幅扰动](#6-phase-2)
7. [Phase 3：单时间窗全频段掩码（新增）](#7-phase-3)
8. [综合发现与神经科学解释](#8-综合发现)
9. [跨被试方案](#9-跨被试方案)
10. [参考文献](#10-参考文献)

---

## 1. 实验背景与目标

NeuroBridge（Zhang et al., 2025）是当前 EEG-to-Image 解码的最优模型，在 THINGS-EEG2 数据集上取得 Top-1 63.2%（论文报告）、sub-08 pilot 实验 Baseline Top-1 69.0%。本系列消融实验系统回答四个核心问题：

| 问题 | 对应实验 |
|---|---|
| 模型依赖哪个**时间段**的 EEG？ | Phase 1 时间窗 × 频段组合 |
| **哪个频段**最关键？（含 Gamma 细分） | Phase 1 精细化七频段 |
| 信号功率（振幅）和时序（相位）哪个更重要？ | Phase 2 振幅扰动 |
| **哪个时间段的整体信息最关键？** | Phase 3 全频段掩码（新增） |

**实验配置：**

| 参数 | 值 |
|---|---|
| 被试 | sub-08（论文最高准确率被试） |
| 模型权重 | 冻结（不训练） |
| Baseline Top-1 | 69.0%（论文 71.2%，差距 <2.2%） |
| Baseline Top-5 | 94.5%（论文 95.1%） |
| 评估方式 | image_test_aug=True，与论文完全一致 |
| 测试集 | 200 张图像，200-way retrieval |
| 采样率 | 250 Hz（THINGS-EEG2 预处理后） |

---

## 2. 频段划分修订说明

### 2.1 v1 版本的问题

| 脚本 | v1 Gamma 定义 | 问题 |
|---|---|---|
| blation_PSD.py | 30–45 Hz | 丢失 45–100Hz 全部信息，与 temporal 脚本不一致 |
| blation_temporal_stft.py | gamma: 30–80Hz；hi_gamma: 80–120Hz | 两脚本不一致；120Hz 超出数据集物理上限 |

### 2.2 数据集物理约束（THINGS-EEG2）

THINGS-EEG2（Gifford et al., 2022）在线滤波为 **0.1–100 Hz**（BrainVision actiCHamp）。NeuroBridge 使用 250Hz 采样率版本，Nyquist 上限 125Hz，但在线滤波已硬截止 100Hz。

- **100Hz 是频段划分的绝对上限**
- **80–100Hz 的 scalp EEG 信噪比极低**（肌电伪迹 EMG 在 20–300Hz 主导）
- 原 hi_gamma: 80–120Hz 中 100–120Hz 部分为数值噪声，**该设定无效，已删除**

### 2.3 v2-B 统一频段方案

依据：Fries (2015) CTC 理论 + Niedermeyer & da Silva 教科书 + THINGS-EEG2 采集规格

| 频段 | 范围 | 神经科学功能 | 对应 v1 |
|---|---|---|---|
| delta | 1–4 Hz | 慢波整合，晚期评价 | 不变 |
| theta | 4–8 Hz | 跨区域整合，注意力采样 | 不变 |
| alpha | 8–13 Hz | **抑制性门控**，视觉皮层抑制（Fries 2015, p.223） | 不变 |
| beta | 13–30 Hz | **自上而下反馈**，预测信号（Fries 2015, p.230） | 不变 |
| **low_gamma** | **30–45 Hz** | 经典 Gamma 振荡下端，视觉特征绑定起始 | 原 PSD 脚本的 gamma |
| **gamma** | **45–70 Hz** | 核心视觉 Gamma，V1/V2 前馈响应峰值区域 | 原 temporal gamma 前半段 |
| **high_gamma** | **70–100 Hz** | 宽带 Gamma 高端；scalp SNR 低，需谨慎解读 | 替代原 hi_gamma (80–120Hz) |

**边界选取依据：**

| 边界 | 依据 |
|---|---|
| 30 Hz | Beta/Gamma 经典分界，Fries 2015，神经科学共识 |
| 45 Hz | Bhattacharyya (BCI)、情绪识别研究的 low-gamma 上界；历史上 gamma 曾被定义为 30–45Hz |
| 70 Hz | 接近视觉皮层前馈 gamma 峰值（约60Hz）右侧；高 gamma 的自然分界 |
| 100 Hz | THINGS-EEG2 数据集在线滤波硬截止（Gifford et al., 2022 §2.2） |

---

## 3. 时间窗划分

时间窗保持 v1 不变（依据 Cichy et al. 2014；Thorpe et al. 1996）：

| 名称 | 采样点范围 | 精确时间（250Hz） | 神经科学依据 |
|---|---|---|---|
| T0 | 0–13 pts | 0–52 ms | 刺激极早期，V1 尚未响应 |
| T1 | 13–38 pts | **52–152 ms** | V1 前馈峰值（Cichy 2014: V1 峰 101ms） |
| T2 | 38–75 pts | **152–300 ms** | N170/类别解码峰值（Thorpe 1996: 分化起点 152ms） |
| T3 | 75–125 pts | 300–500 ms | P300，注意力整合 |
| T4 | 125–200 pts | 500–800 ms | 晚期认知加工 |
| T_full | 0–250 pts | 0–1000 ms | 全段对照 |

> **注意**：250Hz 采样率下 1点=4ms，T1 精确为 52–152ms；报告标注 '~50–150ms' 以对齐神经科学先验。

---

## 4. 实验设计总览（三阶段）

`
Phase 1 ── STFT 时频掩码（7频段 × 5时间窗 = 35组 + 4对照 = 39次推理）
            取 Top-3 关键组合 →
Phase 2 ── 振幅扰动（3类扰动 × 3条件 × 多参数 = 61次推理）

Phase 3 ── 单时间窗全频段掩码（5时间窗 + 1基准 = 6次推理）【新增，独立实验】
`

**总推理次数（v2-B）：** 39 + 61 + 6 = **106 次**（v1 为 86 次）

---

## 5. Phase 1：STFT 时频掩码（精细化 Gamma）

### 5.1 优先级矩阵（7频段 × 5时间窗）

| 时间窗 | delta | theta | alpha | beta | **low_gamma** | **gamma** | **high_gamma** |
|---|---|---|---|---|---|---|---|
| T0 (0–52ms) | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
| T1 (52–152ms) | 1 | 1 | 2 | 3 | ★5 | ★5 | 3 |
| T2 (152–300ms) | 1 | 2 | 2 | 4 | ★5 | ★5 | 3 |
| T3 (300–500ms) | 1 | 3 | 3 | 2 | 2 | 2 | 2 |
| T4 (500–800ms) | 1 | 2 | 2 | 1 | 1 | 1 | 1 |

优先级：★5=最高（必做），4=高，3=中，2=低，1=可选

### 5.2 v2-B 新增关键组合（Gamma 精细化）

| 条件名 | 时间窗 | 频段范围 | 科学问题 |
|---|---|---|---|
| T1_50-150ms__low_gamma | T1 (52–152ms) | 30–45 Hz | Gamma 低端 V1 前馈贡献 |
| T1_50-150ms__gamma | T1 (52–152ms) | 45–70 Hz | **核心 Gamma**（Fries 2015 前馈峰值区域） |
| T1_50-150ms__high_gamma | T1 (52–152ms) | 70–100 Hz | 高端 Gamma（scalp SNR 低，EMG 伪迹区） |
| T2_150-300ms__low_gamma | T2 (152–300ms) | 30–45 Hz | 类别解码时窗的低 Gamma |
| T2_150-300ms__gamma | T2 (152–300ms) | 45–70 Hz | **类别解码时窗的核心 Gamma** |
| T2_150-300ms__high_gamma | T2 (152–300ms) | 70–100 Hz | 类别解码时窗的高 Gamma |

### 5.3 对照组（4个）

| 条件 | 操作 | 目的 |
|---|---|---|
| baseline | 不处理 | 基准线 |
| full_mask_all | 全段 × 全频置零 | 验证频率信息总体必要性 |
| full_time_gamma | 全段 × gamma(45–70Hz) | 验证 T1/T2 gamma 的时间特异性 |
| random_control | 随机低优先级条件 | 验证实验无随机误差 |

---

## 6. Phase 2：振幅扰动

对 Phase 1 Top-3 组合施加三类扰动（方案与 v1 相同，目标条件根据 v2-B 结果更新）：

| 扰动类型 | 操作 | 参数档位 | 核心问题 |
|---|---|---|---|
| **振幅缩放** | STFT 振幅 × α | α = 0.0/0.1/0.2/0.4/0.6/0.8/1.0/1.5/2.0（9档） | 对绝对功率的敏感性；α=0.6–0.8 模拟 AD 振幅退化 |
| **相位随机化** | 随机化相位 | rand = 0%/25%/50%/75%/100%（5档） | 模型依赖精确时序（CTC 机制）吗？ |
| **高斯噪声注入** | 添加噪声至目标 SNR | SNR = +20/+10/+5/0/-5/-10 dB（6档） | 实际 BCI 噪声环境下的鲁棒性 |

---

## 7. Phase 3：单时间窗全频段掩码（新增）

### 7.1 设计动机

Phase 1 每次掩码一个时间窗内的单一频段，回答「哪个时频点最关键」。
Phase 3 每次将某时间窗内**所有频段同时置零**，回答「**哪个时间段承载了整体解码信息**」。

> **核心假设检验**：若将 T2（152–300ms）全频段 STFT 同时置零后准确率下降最大，
> → 说明 T2 时间窗是 NeuroBridge 解码的核心时段，多频段协同编码。

### 7.2 实验条件（6个）

| 条件名 | 操作 | 时间范围 | 科学目的 |
|---|---|---|---|
| ull_freq_T0 | T0 全频置零 | 0–52 ms | 极早期是否干扰？（预测：无影响） |
| ull_freq_T1 | T1 全频置零 | 52–152 ms | V1 前馈整体贡献 |
| **ull_freq_T2** | **T2 全频置零** | **152–300 ms** | **N170/类别解码整体贡献（核心假设）** |
| ull_freq_T3 | T3 全频置零 | 300–500 ms | P300 整体贡献 |
| ull_freq_T4 | T4 全频置零 | 500–800 ms | 晚期加工（预测：负值，去掉有益） |
| aseline | 不处理 | — | 对照基准 |

**实现方式**：对目标时间窗内的全部 STFT 复数系数设为 0 → ISTFT 重建，等价于将该时段 EEG 替换为接近直流的平坦信号，抹去该段内所有振荡信息。

### 7.3 关键预测

| 条件 | 预测 Δ Top-1 | 依据 |
|---|---|---|
| full_freq_T2 | **最大**（约 +0.30 以上） | Thorpe 1996 分化起点 152ms；THINGS-EEG2 解码峰值 |
| full_freq_T1 | 第二（约 +0.20–0.25） | Cichy 2014 V1 峰值 101ms |
| full_freq_T3 | 第三（约 +0.10–0.15） | P300 + 注意力整合 |
| full_freq_T0 | 最小（约 +0.05） | 刺激前，几乎无类别信息 |
| full_freq_T4 | 可能负值 | v1：T4×gamma Δ=−0.020，晚期 gamma 是噪声 |

### 7.4 与 Phase 1 的比较框架

`
Phase 1（单频段掩码）：
  T2×beta   Δ Top-1 = +0.110
  T1×alpha  Δ Top-1 = +0.185
  T2×gamma  Δ Top-1 = +0.005（几乎无贡献）

Phase 3（全频段掩码）：
  full_freq_T2  Δ Top-1 = +0.455（实测，核心假设完整验证）

关键问题：若 full_freq_T2 >> 各单频段 T2×X 之和，
  → T2 时窗内多频段协同效应（非线性叠加）
  → 不能简化为单频段 beta 的独立贡献
`

### 7.5 实验结果（已完成，Sub-08 Pilot）

| 条件 | Top-1 | Top-5 | Δ Top-1 | Δ Top-5 | mean_rank |
|---|---|---|---|---|---|
| baseline | 0.690 | 0.945 | 0.000 | 0.000 | 2.055 |
| full_freq_T0 | 0.700 | 0.950 | **-0.010** | -0.005 | 2.080 |
| full_freq_T1 | 0.425 | 0.735 | **+0.265** | +0.210 | 6.835 |
| **full_freq_T2** | **0.235** | **0.470** | **+0.455** | **+0.475** | **22.14** |
| full_freq_T3 | 0.575 | 0.840 | +0.115 | +0.105 | 3.835 |
| full_freq_T4 | 0.610 | 0.890 | +0.080 | +0.055 | 3.015 |

---

## 8. 综合发现与神经科学解释

### 8.1 v1 核心发现（待用 v2-B 精细化 Gamma 验证）

| 发现 | v1 结果 | v2-B 验证目的 |
|---|---|---|
| **Gamma 不是最重要的** | T1×gamma(30–80Hz) 仅排第 9（Δ=+0.015） | 细分三档：确认哪段贡献最强 |
| **Alpha 是负向特征** | T1×alpha Δ=+0.185（最大），振幅越大准确率越低 | v2-B 对照验证 |
| **T2×beta 相位携带类别信息** | 相位 Δ=+0.155 > 振幅 Δ=+0.110 | 保持不变 |
| **AD 振幅退化对 T2×beta 无影响** | α=0.6–0.8 区间 Δ≈0 | 保持不变 |

### 8.2 三档 Gamma 细化的预期神经科学解释

**情形 A：gamma(45–70Hz) 最显著（最优假设）**  
符合 Fries (2015) CTC 理论：视觉前馈 Gamma 峰值约在 50–70Hz，提示 NeuroBridge 提取了视觉皮层前馈 Gamma 同步信息。

**情形 B：三档 Gamma 均无显著贡献（与 v1 一致）**  
NeuroBridge 解码依赖 alpha/beta 调制性信号，而非感觉皮层 Gamma 前馈。
提示模型学到的是视觉**认知状态**表征（反直觉发现，具有发表价值）。

**情形 C：high_gamma(70–100Hz) 阴性（预测最可能）**  
直接引用 Niedermeyer & da Silva (2004) 关于 scalp EEG 高频 EMG 伪迹的说明，增强结果可信度。

### 8.3 Phase 3 的神经科学意义

若 ull_freq_T2 产生最大 Drop（且远大于单频段 T2×beta）：
- T2（152–300ms）时窗内**多频段协同编码**了关键解码信息
- 对应 Thorpe (1996) 提出的 152ms 视觉分类分化时间窗
- 单一频段分析无法捕捉该时窗的整体重要性，Phase 3 提供不可或缺的补充证据

---

## 9. 跨被试方案

| 优先级 | 被试 | 理由 |
|---|---|---|
| ✅ 已完成 | sub-08 | Pilot，最高准确率 |
| 1 | sub-04 | 高准确率，结果可信 |
| 2 | sub-07 | 高准确率 |
| 3 | sub-10 | 中等偏高 |
| 4 | sub-03/06/09 | 中等准确率 |
| 5 | sub-01/02/05 | 低准确率，最后跑 |

**Phase 3 特别关注**：ull_freq_T2 的跨被试 Δ Top-1 一致性。若跨被试稳定，
可在论文 Results 中报告：「T2（152–300ms）是 NeuroBridge 解码的必要时间窗（跨被试一致）」

---

## 10. 参考文献

| 编号 | 文献 | 本版本的作用 |
|---|---|---|
| [1] | Gifford et al. (2022). *A large and rich EEG dataset for modeling human visual object recognition*. NeuroImage 264, 119754. | 数据集规格；100Hz 上限；时间窗 |
| [2] | Zhang et al. (2025). *NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding*. | 模型架构，Baseline 准确率 |
| [3] | Fries (2015). *Rhythms for Cognition: Communication through Coherence*. Neuron 88, 220–235. | 三档 Gamma 边界（前馈 Gamma 峰值约 45–70Hz） |
| [4] | Thorpe, Fize & Marlot (1996). *Speed of processing in the human visual system*. Nature 381, 520–522. | T2 时间窗（152ms 分化起点） |
| [5] | Cichy, Pantazis & Oliva (2014). *Resolving human object recognition in space and time*. Nat Neurosci 17, 455–462. | T1 时间窗（V1峰 101ms） |
| [6] | Niedermeyer & da Silva (2004). *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields*. | Gamma 上限标准；scalp EMG 伪迹说明 |
| [7] | Bhattacharyya et al. (2019). *A multi-channel architecture for adaptive EEG-based motor imagery classification*. IEEE Trans Neural Syst Rehabil Eng. | 30–45Hz low_gamma 边界（BCI 领域） |
| [8] | Haufe et al. (2014). *On the interpretation of weight vectors of linear models in multivariate neuroimaging*. NeuroImage 87, 96–110. | Phase 2 振幅扰动解释框架 |
| [9] | Wang et al. (2026). *FourierMask: Explain EEG-Based End-to-End Deep Learning Models in the Frequency Domain*. IEEE JBHI 30(4). | STFT 掩码方法论 |

---

*本报告对应脚本：blation_temporal_stft.py（v2-B，含 Phase 3 全频段掩码）及 blation_temporal_amplitude.py（v2-B）。*
*上一版本：NeuroBridge_Temporal_Ablation_Report.md（v1，gamma: 30–80Hz，hi_gamma: 80–120Hz）。*
