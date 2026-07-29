# NeuroBridge Frequency-Band Zero Ablation (sub-08)

Frequency-band Fourier-mask zeroing on Things-EEG **sub-08**, evaluated with the official NeuroBridge EEG-to-image checkpoint (`EEGProject` + linear projectors).

Related project notes live in this repository ([2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-)).

## Results (Top-1 / Top-5)

Eval settings matched `intra-subjects.sh`: `--image_test_aug`, RN50 aug features, `--img_l2norm`, `seed=2025`.

| Condition | Top-1 (%) | Top-5 (%) | Δ Top-1 |
|-----------|-----------|-----------|---------|
| original | 69.00 | 94.50 | 0.00 |
| zero delta | 49.00 | 77.00 | -20.00 |
| zero theta | 46.00 | 73.00 | -23.00 |
| zero alpha | 46.50 | 78.50 | -22.50 |
| zero beta | 46.00 | 78.00 | -23.00 |
| zero low_gamma | 67.50 | 95.00 | -1.50 |
| zero gamma | 69.00 | 95.00 | 0.00 |
| zero high_gamma | 69.00 | 95.50 | 0.00 |

Files:

- `results/sub-08_ori_vs_zero_acc.csv`
- `results/sub-08_ori_vs_zero_acc.png`
- `checkpoints/intra-subjects_sub-08_checkpoint_last.pth`

## Code

| File | Role |
|------|------|
| `code/simply_erase_the_band.py` | Fourier-mask zero selected bands → save new `train.npy` / `test.npy` |
| `code/acc_zero-bands` | Load checkpoint; compare original vs each zeroed band (Top-1/Top-5) |

Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100.

## Why EEG `.npy` data are NOT in this GitHub repo

Each `sub-08` folder is about **4.8 GB** (train+test). Seven zeroed bands ≈ **34 GB**, which exceeds normal GitHub / free Git LFS limits.

**Regenerate locally** (inside a NeuroBridge checkout with preprocessed EEG):

```powershell
conda activate neurobridge
cd <NeuroBridge-root>
$env:PYTHONPATH = "."

# example: zero alpha for all subjects (or add --sub_ids 8)
python "simply erase the band" --bands alpha --sub_ids 8

# evaluate
python data/acc_zero-bands
```

Expected zeroed data layout:

```text
data/things_eeg/preprocessed_eeg_zero_<band>_fourier/sub-08/{train,test}.npy
```

If you host the `.npy` packs elsewhere (Hugging Face / NetDisk), put the download URL here:

- **Data download:** _(TODO: add link)_

## Re-run evaluation only

```powershell
conda activate neurobridge
cd <NeuroBridge-root>   # needs module/, image features, EEG dirs
$env:PYTHONPATH = "."
python NeuroBridge_freq_band_zero/code/acc_zero-bands `
  --checkpoint_path NeuroBridge_freq_band_zero/checkpoints/intra-subjects_sub-08_checkpoint_last.pth `
  --eeg_ori_dir data/things_eeg/preprocessed_eeg `
  --zero_root data/things_eeg `
  --device cpu
```

(Or copy `acc_zero-bands` back under `data/` as in the original NeuroBridge layout.)
