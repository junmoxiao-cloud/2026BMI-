# 🚀 Call for Collaborators: 基于 EEG-BCI 的阿尔茨海默症早期检测与干预研究

**项目名称：** Toward Real-time EEG Cognitive Feedback Therapy for the MCI/AD Continuum
**共同发起人：** 沈旎（上海交通大学 计算机科学与技术 24级）乔钰成（华东师范大学 计算机科学与技术 25级） 
**指导教授：** Dr. Neal Bangerter（帝国理工学院 生物工程系 副教授）
**联系方式：** dou12345622@qq.com

---

## 📌 探索方向 (Research Directions)

本项目依托静息态脑电图（Resting-state EEG），主要探索两个互补的研究方向：

1. **方向一：EEG 生物标志物挖掘 (Biomarker Extraction)**
   基于公开数据集，提取 EEG 微状态（Microstate）的时间动态参数与频谱特征，探索其在“健康-主观认知下降(SCD)-轻度认知障碍(MCI)”连续体中的演变规律，并利用机器学习（SVM、随机森林等）进行分类预测。
2. **方向二：闭环神经反馈系统 (Closed-Loop Neurofeedback System)**
   构建一套实时 BCI 系统，将提取到的异常微状态特征作为反馈信号。通过视觉/听觉反馈（Operant Conditioning），训练使用者主动调节自身大脑状态，向健康模式靠拢。

---

## 💡 理由与应用价值 (Rationale & Application Value)

- **临床痛点与价值**：SCD 和 MCI 是阿尔茨海默症干预的黄金窗口期。目前缺乏低成本的早期筛查手段和有效的非药物干预疗法。本研究致力于填补这一空白。
- **为何选择 EEG？** 相比于 fMRI 或 PET，EEG 具有**极低的成本、非侵入性**以及**毫秒级的时间分辨率**，非常适合在社区或家庭环境中进行大规模的早期筛查与日常认知训练。
- **开源与可复现**：本项目全程采用开源工具栈（如 MNE-Python, OpenBCI），致力于打造高透明度、易于复现的科研成果。

---

## 🗓️ 极简执行规划 (2-Month Fast-Track Plan)

为保证高效产出（目标投递 IEEE EMBC 或 IEEE NER），我们采取“短平快”的敏捷科研节奏：

- **第 1 个月：核心实验与数据处理 (Experiment)**
  - 跑通 MNE-Python 预处理 pipeline。
  - 在公开数据集（如 OpenNeuro ds004504）上完成微状态聚类与特征提取。
  - 训练分类模型并产出核心图表与数据结果。
- **第 2 个月：论文撰写与打磨 (Writing)**
  - 汇总数据结果，完成统计学检验。
  - 集中撰写论文（Abstract, Intro, Methods, Results, Discussion）。
  - 内部精修定稿并完成目标会议投稿。

---

如果你对 **机器学习、脑机接口（BCI）或神经工程** 感兴趣，且希望在高效的节奏下产出高质量的学术成果，欢迎与我联系组队！
