开始前至少确认：

- \(K=4\) 模板来自pre-MVNN训练集；
- 4个状态都有合理coverage；
- 模板间没有明显重复；
- 模板图没有明显伪迹；
- 固定最终的 `microstate_templates.npy` 和状态编号。

之后按以下流程实施：

```text
pre-MVNN test single trials
→ 使用固定K=4模板back-fitting
→ 得到每个时间点的microstate标签
→ 扰动目标状态对应的连续EEG片段
→ 使用训练集得到的固定MVNN矩阵进行白化
→ 按80个repetitions平均
→ 选择原来的17个通道
→ 输入sub-08 checkpoint
→ 比较Top-1和Top-5
```

建议分三个阶段做。

### 1. 先重建无消融baseline

这是最重要的验证：

```text
pre-MVNN test EEG
→ 不做任何microstate修改
→ 原始MVNN
→ repetition averaging
→ 17通道
→ NeuroBridge
```

该结果应该与官方MVNN版 `test.npy` 以及 `ablation_PSD.py` clean baseline基本一致。如果差异很大，先不要进行microstate消融，因为这说明MVNN重建、session顺序、trial averaging或通道顺序不一致。

### 2. 实现最简单的状态消融

第一轮建议不要同时修改duration、occurrence和transition，而是对每个状态进行统一的segment replacement：

- 找到目标状态的全部连续片段；
- 使用匹配长度的非目标状态片段或受控噪声替换；
- 保持数组形状始终为 `63 × 250`；
- 分别消融state 1、2、3、4；
- 每种条件使用相同随机种子；
- 记录实际修改的采样点比例。

必须添加matched random-time control。例如state 2覆盖6.3%，那么随机对照也修改每个trial中约6.3%的时间点。否则state 1/4造成更大性能下降，可能只是因为修改的数据更多。

第一轮应至少包含：

| 条件 | 作用 |
|---|---|
| Clean baseline | 无修改 |
| State 1 ablation | 消融状态1 |
| State 2 ablation | 消融状态2 |
| State 3 ablation | 消融状态3 |
| State 4 ablation | 消融状态4 |
| Matched random-time | 修改相同比例随机时间点 |
| Matched contiguous-segment | 修改相同数量和长度的随机连续片段 |

### 3. 验证消融真的实现了目标

每次扰动后重新使用固定模板back-fit，并报告：

- 目标状态coverage变化；
- occurrence变化；
- mean duration变化；
- transition probability变化；
- mean GFP变化；
- EEG RMS或总功率变化；
- 实际修改采样点比例。

因为替换一个状态片段通常会同时改变多个microstate特征，所以不能只声称“消融了duration”而不检查其他指标。

另外，后续MVNN必须满足：

\[
W_s=\Sigma_{\mathrm{train},s}^{-1/2}
\]

每个session使用原训练数据计算的固定矩阵。所有clean和ablation条件共用同一个 \(W_s\)，不能从扰动后的测试数据重新估计。

因此结论是：你现在可以开始编写消融代码；第一项任务应当是保存每个session的MVNN矩阵并成功重建clean baseline。只有这个baseline与原模型输入和准确率对齐后，microstate消融造成的性能差异才能被可信地解释。

已经按照你现在的两阶段实验规划修改了 [microstate_ablation_reference.py](D:/shenni/交大/论文/EEG2Image/NeuroBridge/microstate_ablation_reference.py)。

当前支持：

```text
clean
topography
duration
occurrence
amplitude
```

并支持：

```text
none
random-time
random-segment
```

三种对照设置。

代码已经通过语法和合成数据测试，确认：

- 所有干预保持 `63×250`；
- topography、duration和occurrence替换保持瞬时GFP；
- amplitude的GFP缩放比例准确；
- random-time control修改的采样点数量与目标干预一致。

## Phase 1：分别消融4个microstate模板

完整替换目标状态出现位置的空间拓扑：

```powershell
python microstate_ablation_reference.py `
  --condition topography `
  --state 0 `
  --control none `
  --output-dir "results\microstate_ablation_sub08"
```

分别运行：

```text
--state 0：State 1
--state 1：State 2
--state 2：State 3
--state 3：State 4
```

也保留了旧名称兼容：

```powershell
--condition state
```

等同于：

```powershell
--condition topography
```

### Matched random-time control

例如State 1：

```powershell
python microstate_ablation_reference.py `
  --condition topography `
  --state 0 `
  --control random-time `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

### Matched random-segment control

```powershell
python microstate_ablation_reference.py `
  --condition topography `
  --state 0 `
  --control random-segment `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

Phase 1结束后比较：

\[
\Delta Acc_{\mathrm{specific},k}
=
\Delta Acc_{\mathrm{topography},k}
-
\Delta Acc_{\mathrm{control},k}
\]

由此确定哪个microstate模板对模型最重要。

## Phase 2：分析重要状态的特征

假设Phase 1发现State 2最重要，则使用：

```text
--state 1
```

### 1. Duration/Coverage缩短

例如将每个符合条件的State 2 episode缩短25%：

```powershell
python microstate_ablation_reference.py `
  --condition duration `
  --state 1 `
  --ratio 0.25 `
  --control none `
  --output-dir "results\microstate_ablation_sub08"
```

脚本会：

1. 找到每个连续State 2 episode；
2. 找到它后面的microstate；
3. 将State 2末端25%的拓扑替换为后续状态模板；
4. 保留至少20 ms的State 2片段；
5. 排除没有following state的trial末端episode；
6. 保持修改时间点的GFP不变。

建议测试：

```text
--ratio 0.10
--ratio 0.25
--ratio 0.50
```

不建议duration直接使用100%，因为这会把episode完全删除，变成occurrence实验。

对应随机片段对照：

```powershell
python microstate_ablation_reference.py `
  --condition duration `
  --state 1 `
  --ratio 0.25 `
  --control random-segment `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

## 2. Occurrence removal

随机移除25%的完整State 2 episodes：

```powershell
python microstate_ablation_reference.py `
  --condition occurrence `
  --state 1 `
  --ratio 0.25 `
  --control none `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

每个被选中的episode会：

1. 作为完整连续片段被处理；
2. 计算其与其他3个模板的平均绝对空间相关；
3. 选择episode层面的第二相似microstate；
4. 整个episode使用同一个替代状态；
5. 保留每个时间点的GFP。

建议剂量：

```text
--ratio 0.25
--ratio 0.50
--ratio 0.75
--ratio 1.00
```

其中 `1.00` 表示替换该状态全部episode。

因为episode选择具有随机性，每个比例建议运行多个seed：

```text
2025
2026
2027
2028
2029
```

例如：

```powershell
python microstate_ablation_reference.py `
  --condition occurrence `
  --state 1 `
  --ratio 0.50 `
  --control none `
  --random-state 2026 `
  --output-dir "results\microstate_ablation_sub08"
```

对应matched control：

```powershell
python microstate_ablation_reference.py `
  --condition occurrence `
  --state 1 `
  --ratio 0.50 `
  --control random-segment `
  --random-state 2026 `
  --output-dir "results\microstate_ablation_sub08"
```

## 3. GFP amplitude

将State 2的GFP缩小至原来的50%：

```powershell
python microstate_ablation_reference.py `
  --condition amplitude `
  --state 1 `
  --gfp-scale 0.5 `
  --control none `
  --output-dir "results\microstate_ablation_sub08"
```

建议测试：

```text
--gfp-scale 0.25
--gfp-scale 0.50
--gfp-scale 0.75
--gfp-scale 1.00
--gfp-scale 1.25
--gfp-scale 1.50
```

其中：

```text
1.00 = clean amplitude
<1.00 = 降低GFP
>1.00 = 放大GFP
```

对应随机时间对照：

```powershell
python microstate_ablation_reference.py `
  --condition amplitude `
  --state 1 `
  --gfp-scale 0.5 `
  --control random-time `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

## 输出文件

新版本使用：

```text
results/microstate_ablation_sub08/
└── microstate_feature_ablation_results.csv
```

避免与旧版 `retrieval_results.csv` 混合。

每个条件还会生成独立JSON，例如：

```text
topography_state-1_full_control-none_seed-2025.json
duration_state-2_ratio-0.25_control-none_seed-2025.json
occurrence_state-2_ratio-0.5_control-random-segment_seed-2026.json
amplitude_state-2_scale-0.5_control-none_seed-2025.json
```

结果包含：

- 实际修改采样点比例；
- eligible episode数量；
- selected episode数量；
- 实际episode选择比例；
- Top-1和Top-5；
- accuracy drop；
- 消融后coverage；
- occurrence；
- duration；
- transition matrix；
- mean GFP；
- mean \(|r|\)；
- GEV；
- RMS比例。

## Transition暂未加入

当前主脚本没有加入block shuffle，因为它会同时改变：

- microstate顺序；
- 刺激后时间位置；
- ERP latency；
- 波形连续性；
- 边界频谱。

因此，现阶段更合理的顺序是：

```text
Phase 1：State 1–4空间模板消融
Phase 2：重要状态的duration、occurrence和GFP剂量实验
Phase 3：将transition作为探索性实验单独实现
```

这与当前结果图规划一致：上方比较4个模板的状态特异消融，下方针对最重要状态画25%–100%的occurrence dose-response，或者duration/GFP的剂量曲线。s