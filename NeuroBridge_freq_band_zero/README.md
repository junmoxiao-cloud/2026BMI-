# NeuroBridge Frequency-Band Ablation (sub-08)

Things-EEG **sub-08** frequency-band experiments with the official NeuroBridge EEG-to-image checkpoint (`EEGProject` + linear projectors).

Repo: [2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-)

## Pipeline

1. **Zero band** (Fourier mask) → band-zeroed EEG  
2. **Extract** original band amplitude & phase (`rFFT`)  
3. **Replace** zeroed bins with power-matched complex Gaussian noise  
4. **Evaluate** Top-1 / Top-5: original vs zero vs noise-replace  

Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100.

## Results (Top-1 / Top-5)

Eval matched `intra-subjects.sh`: `--image_test_aug`, RN50 fused aug features, `--img_l2norm`, `seed=2025`, device CPU.

Local baseline for this `checkpoint_last` is **69.0%** Top-1 (not the paper’s ~71.2%; same settings, this weight).

### Original vs zero vs noise-replace

| Condition | Top-1 (%) | Top-5 (%) | Δ Top-1 |
|-----------|-----------|-----------|---------|
| original | 69.00 | 94.50 | 0.00 |
| zero delta | 49.00 | 77.00 | −20.00 |
| noise delta | 52.00 | 79.50 | −17.00 |
| zero theta | 46.00 | 73.00 | −23.00 |
| noise theta | 41.00 | 69.50 | −28.00 |
| zero alpha | 46.50 | 78.50 | −22.50 |
| noise alpha | 49.50 | 80.00 | −19.50 |
| zero beta | 46.00 | 78.00 | −23.00 |
| noise beta | 46.50 | 76.50 | −22.50 |
| zero low_gamma | 67.50 | 95.00 | −1.50 |
| noise low_gamma | 66.00 | 94.00 | −3.00 |
| zero gamma | 69.00 | 95.00 | 0.00 |
| noise gamma | 68.50 | 94.50 | −0.50 |
| zero high_gamma | 69.00 | 95.50 | 0.00 |
| noise high_gamma | 68.50 | 94.50 | −0.50 |

Files:

- `results/sub-08_ori_zero_noise_acc.csv` / `.png` — three-way comparison  
- `results/sub-08_ori_vs_zero_acc.csv` / `.png` — original vs zero only  
- `checkpoints/intra-subjects_sub-08_checkpoint_last.pth`

## Code

| File | Role |
|------|------|
| `code/simply_erase_the_band.py` | Fourier-mask zero selected bands |
| `code/amplitude_and_phase_original` | Extract per-band amp/phase from original EEG |
| `code/gaussian-noise` | Replace zeroed bins with power-matched Gaussian noise |
| `code/acc_zero-bands` | Eval original vs each zeroed band |
| `code/acc_ori_zero_noise` | Eval original vs zero vs noise-replace |

Noise power match: \(N = N_0\sqrt{P_S / P_{N_0}}\) with \(P=\mathrm{mean}(|\cdot|^2)\), and \(\mathrm{Re}/\mathrm{Im}(N_0)\sim\mathcal N(\mu,\sigma)\) from original \(S=A e^{j\phi}\).

## Why large EEG `.npy` are NOT in this repo

Each sub-08 train+test pack is ~4–5 GB; seven zeroed + seven noise-replaced packs ≫ GitHub limits.

**Regenerate locally** inside a NeuroBridge checkout:

```powershell
conda activate neurobridge
cd <NeuroBridge-root>
$env:PYTHONPATH = "."

# 1) zero bands
python "simply erase the band" --bands alpha --sub_ids 8

# 2) amp/phase from original
python "data/amplitude&phase(orignal)"

# 3) noise replace into zeroed EEG
python data/gaussian-noise

# 4) evaluate
python data/acc_ori_zero_noise --device cpu
```

Expected local layouts:

```text
data/things_eeg/sub-08-zero-bands/sub-08-<band>/{train,test}.npy
data/amplitude&phase(orignal)_out/sub-08/{train,test}/
data/gaussian-noise_out/sub-08/<band>/{train,test}.npy
```

## Re-run evaluation only

```powershell
conda activate neurobridge
cd <NeuroBridge-root>
$env:PYTHONPATH = "."
python NeuroBridge_freq_band_zero/code/acc_ori_zero_noise `
  --checkpoint_path NeuroBridge_freq_band_zero/checkpoints/intra-subjects_sub-08_checkpoint_last.pth `
  --eeg_ori_dir data/things_eeg/preprocessed_eeg `
  --device cpu
```
