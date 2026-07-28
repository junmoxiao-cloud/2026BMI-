"""
verify_ablation_logic.py
========================
完整验证脚本：不依赖 NeuroBridge 模型/数据，纯用合成 EEG 验证
ablation_temporal_stft.py 的信号处理逻辑是否正确。

验证维度
--------
T1  STFT 掩码定向性  : 被掩码频段能量应 ≈ 0；非目标频段能量不应变化
T2  掩码时间局部性   : 目标时间窗外的信号保持原样
T3  ISTFT 重建精度   : 无实质掩码时 STFT->ISTFT 重建误差 < 0.05
T4  降级路径（FFT）  : 时间窗 < nperseg 时正确切换到直接 FFT 掩码
T5  全频带置零       : mask_all_bands 后信号近似 DC
T6  Phase3 全频掩码  : stft_mask_full_freq_window 后目标窗段 AC 能量 ≈ 0，DC 保留
T7  STFT 时频图可视化: 对合成 multi-freq 信号绘制"掩码前 vs 掩码后"对比图
T8  v2b 三分 Gamma   : low_gamma/gamma/high_gamma 各自定向消除

输出
----
  verify_stft_spectrogram_before.png   -- 掩码前 STFT 时频图
  verify_stft_spectrogram_after.png    -- 掩码后 STFT 时频图
  verify_stft_spectrogram_compare.png  -- 并排对比 + 差异图
  verify_ablation_logic_report.txt     -- 文字版测试报告
"""

from __future__ import annotations
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8','utf-8-sig'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft, istft as scipy_istft

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results" / "verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS           = 250.0
STFT_NPERSEG = 32
STFT_NOVERLAP= 16
STFT_NFFT    = 64

TIME_WINDOWS = {
    "T0_0-50ms":    (0,   13),
    "T1_50-150ms":  (13,  38),
    "T2_150-300ms": (38,  75),
    "T3_300-500ms": (75,  125),
    "T4_500-800ms": (125, 200),
    "T_full":       (0,   250),
}
FREQ_BANDS_V1 = {
    "delta":    (1.0,  4.0),
    "theta":    (4.0,  8.0),
    "alpha":    (8.0,  13.0),
    "beta":     (13.0, 30.0),
    "gamma":    (30.0, 80.0),
    "hi_gamma": (80.0, 120.0),
}
FREQ_BANDS_V2 = {
    "delta":      (1.0,   4.0),
    "theta":      (4.0,   8.0),
    "alpha":      (8.0,   13.0),
    "beta":       (13.0,  30.0),
    "low_gamma":  (30.0,  45.0),
    "gamma":      (45.0,  70.0),
    "high_gamma": (70.0, 100.0),
}


def stft_mask_time_freq(eeg, time_window, freq_band, fs=FS,
                        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, nfft=STFT_NFFT):
    eeg = np.asarray(eeg, dtype=np.float32)
    original_shape = eeg.shape
    t_start, t_end = time_window
    low_hz, high_hz = freq_band
    flat = eeg.reshape(-1, eeg.shape[-1])
    ablated_flat = flat.copy()
    segment = flat[:, t_start:t_end]
    if segment.shape[-1] < nperseg:
        fft_seg = np.fft.rfft(segment, axis=-1)
        freqs_fft = np.fft.rfftfreq(segment.shape[-1], d=1.0/fs)
        freq_mask_fft = (freqs_fft >= low_hz) & (freqs_fft <= high_hz)
        fft_seg[:, freq_mask_fft] = 0.0
        ablated_segment = np.fft.irfft(fft_seg, n=segment.shape[-1], axis=-1).astype(np.float32)
        ablated_flat[:, t_start:t_end] = ablated_segment
        return ablated_flat.reshape(original_shape)
    freqs, t_frames, Zxx = scipy_stft(segment, fs=fs, window="hann",
                                       nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1)
    freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
    Zxx_masked = Zxx.copy()
    Zxx_masked[:, freq_mask, :] = 0.0
    _, reconstructed = scipy_istft(Zxx_masked, fs=fs, window="hann",
                                    nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                    time_axis=-1, freq_axis=-2)
    seg_len = t_end - t_start
    rec_len = reconstructed.shape[-1]
    if rec_len >= seg_len:
        reconstructed = reconstructed[:, :seg_len]
    else:
        reconstructed = np.pad(reconstructed, ((0,0),(0, seg_len-rec_len)), mode="edge")
    ablated_flat[:, t_start:t_end] = reconstructed.astype(np.float32)
    return ablated_flat.reshape(original_shape)


def mask_all_bands(eeg, fs=FS):
    eeg = np.asarray(eeg, dtype=np.float32)
    dc = eeg.mean(axis=-1, keepdims=True)
    return np.broadcast_to(dc, eeg.shape).copy()


def stft_mask_full_freq_window(eeg, time_window, fs=FS,
                                nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, nfft=STFT_NFFT):
    eeg = np.asarray(eeg, dtype=np.float32)
    original_shape = eeg.shape
    t_start, t_end = time_window
    flat    = eeg.reshape(-1, eeg.shape[-1])
    segment = flat[:, t_start:t_end]
    if segment.shape[-1] < nperseg:
        dc = segment.mean(axis=-1, keepdims=True)
        ablated = np.broadcast_to(dc, segment.shape).copy().astype(np.float32)
        result = flat.copy(); result[:, t_start:t_end] = ablated
        return result.reshape(original_shape)
    freqs, _, Zxx = scipy_stft(segment, fs=fs, window="hann",
                                nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1)
    Zxx_masked = np.zeros_like(Zxx)
    Zxx_masked[:, 0, :] = Zxx[:, 0, :]
    _, reconstructed = scipy_istft(Zxx_masked, fs=fs, window="hann",
                                    nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                    time_axis=-1, freq_axis=-2)
    seg_len = t_end - t_start
    rec_len = reconstructed.shape[-1]
    if rec_len >= seg_len:
        reconstructed = reconstructed[:, :seg_len]
    else:
        reconstructed = np.pad(reconstructed, ((0,0),(0, seg_len-rec_len)), mode="edge")
    result = flat.copy()
    result[:, t_start:t_end] = reconstructed.astype(np.float32)
    return result.reshape(original_shape)


def make_synthetic_eeg(n_trials=10, n_ch=17, n_tp=250, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n_tp) / FS
    components = [(2.0,0.5),(10.0,0.8),(20.0,0.6),(50.0,1.0),(90.0,0.4)]
    signal = np.zeros(n_tp, dtype=np.float32)
    for freq, amp in components:
        phase = rng.uniform(0, 2*np.pi)
        signal += amp * np.sin(2*np.pi*freq*t + phase).astype(np.float32)
    noise = rng.standard_normal((n_trials, n_ch, n_tp)).astype(np.float32) * 0.05
    return (signal[np.newaxis, np.newaxis, :] + noise).astype(np.float32)


PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"[{status}] {name}"
    if detail: msg += f"\n       -> {detail}"
    print(msg)
    results.append((status, name, detail))
    return condition


def test_t1_freq_selectivity():
    print("\n-- T1: STFT masking frequency selectivity --")
    eeg = make_synthetic_eeg()
    tw = TIME_WINDOWS["T1_50-150ms"]
    fb_gamma = FREQ_BANDS_V1["gamma"]
    ablated = stft_mask_time_freq(eeg, tw, fb_gamma)
    seg_orig = eeg[0, 0, tw[0]:tw[1]]
    seg_abl  = ablated[0, 0, tw[0]:tw[1]]
    def band_energy(sig, low, high, fs=FS):
        f = np.fft.rfftfreq(len(sig), d=1.0/fs)
        S = np.abs(np.fft.rfft(sig))**2
        return S[(f>=low)&(f<=high)].sum()
    gamma_orig = band_energy(seg_orig, 30, 80)
    gamma_abl  = band_energy(seg_abl,  30, 80)
    alpha_orig = band_energy(seg_orig, 8,  13)
    alpha_abl  = band_energy(seg_abl,  8,  13)
    ratio_g = gamma_abl / (gamma_orig + 1e-10)
    ratio_a = alpha_abl / (alpha_orig + 1e-10)
    check("T1a  gamma energy removed (ratio<0.05)",
          ratio_g < 0.05, f"gamma_orig={gamma_orig:.4f}, abl={gamma_abl:.6f}, ratio={ratio_g:.4f}")
    check("T1b  alpha energy intact  (ratio>0.85)",
          ratio_a > 0.85, f"alpha_orig={alpha_orig:.4f}, abl={alpha_abl:.6f}, ratio={ratio_a:.4f}")


def test_t2_time_locality():
    print("\n-- T2: Time locality --")
    eeg = make_synthetic_eeg()
    tw = TIME_WINDOWS["T1_50-150ms"]
    fb = FREQ_BANDS_V1["gamma"]
    ablated = stft_mask_time_freq(eeg, tw, fb)
    rmse_b = np.sqrt(np.mean((eeg[0,0,:tw[0]] - ablated[0,0,:tw[0]])**2))
    rmse_a = np.sqrt(np.mean((eeg[0,0,tw[1]:] - ablated[0,0,tw[1]:])**2))
    check("T2a  region before window unchanged (RMSE<1e-5)", rmse_b < 1e-5, f"RMSE={rmse_b:.2e}")
    check("T2b  region after  window unchanged (RMSE<1e-5)", rmse_a < 1e-5, f"RMSE={rmse_a:.2e}")


def test_t3_reconstruction():
    print("\n-- T3: ISTFT reconstruction fidelity --")
    eeg = make_synthetic_eeg()
    tw  = TIME_WINDOWS["T2_150-300ms"]
    fb_nil = (0.5, 1.0)
    ablated = stft_mask_time_freq(eeg, tw, fb_nil)
    rmse = np.sqrt(np.mean((eeg[0,0,tw[0]:tw[1]] - ablated[0,0,tw[0]:tw[1]])**2))
    check("T3   near-null mask RMSE < 0.05", rmse < 0.05, f"RMSE={rmse:.2e}")


def test_t4_fft_fallback():
    print("\n-- T4: FFT fallback for short window --")
    eeg = make_synthetic_eeg()
    tw_short = TIME_WINDOWS["T0_0-50ms"]
    fb = FREQ_BANDS_V1["alpha"]
    ablated = stft_mask_time_freq(eeg, tw_short, fb)
    seg_orig = eeg[0, 0, tw_short[0]:tw_short[1]]
    seg_abl  = ablated[0, 0, tw_short[0]:tw_short[1]]
    f = np.fft.rfftfreq(len(seg_orig), d=1.0/FS)
    E_orig = np.abs(np.fft.rfft(seg_orig))**2
    E_abl  = np.abs(np.fft.rfft(seg_abl))**2
    m = (f>=8)&(f<=13)
    ratio = E_abl[m].sum() / (E_orig[m].sum() + 1e-10)
    check("T4a  short-win FFT fallback: alpha removed (ratio<0.20)",
          ratio < 0.20, f"ratio={ratio:.4f}")
    check("T4b  output shape preserved", ablated.shape == eeg.shape,
          f"in={eeg.shape}, out={ablated.shape}")


def test_t5_mask_all_bands():
    print("\n-- T5: mask_all_bands -> DC only --")
    eeg = make_synthetic_eeg()
    masked = mask_all_bands(eeg)
    dc_expected = eeg.mean(axis=-1, keepdims=True)
    max_diff = np.abs(masked - dc_expected).max()
    ac_power = np.var(masked, axis=-1).mean()
    check("T5a  output == channel mean", max_diff < 1e-5, f"max_diff={max_diff:.2e}")
    check("T5b  AC variance < 1e-6",    ac_power < 1e-6, f"AC_var={ac_power:.2e}")


def test_t6_phase3():
    print("\n-- T6: Phase3 stft_mask_full_freq_window --")
    eeg = make_synthetic_eeg()
    tw = TIME_WINDOWS["T2_150-300ms"]
    ablated = stft_mask_full_freq_window(eeg, tw)
    seg_o = eeg[0,0,tw[0]:tw[1]]
    seg_a = ablated[0,0,tw[0]:tw[1]]
    ac_o = np.var(seg_o - seg_o.mean())
    ac_a = np.var(seg_a - seg_a.mean())
    ratio = ac_a / (ac_o + 1e-10)
    dc_diff = abs(seg_a.mean() - seg_o.mean())
    rmse_out = np.sqrt(np.mean((eeg[0,0,tw[1]:] - ablated[0,0,tw[1]:])**2))
    check("T6a  target window AC energy < 10%",
          ratio < 0.10, f"AC_orig={ac_o:.4f}, AC_abl={ac_a:.6f}, ratio={ratio:.4f}")
    check("T6b  DC preserved (|diff|<1e-4)",
          dc_diff < 1e-4, f"DC_diff={dc_diff:.2e}")
    check("T6c  outside window unchanged (RMSE<1e-5)",
          rmse_out < 1e-5, f"RMSE_out={rmse_out:.2e}")


def test_t7_visualization():
    print("\n-- T7: STFT spectrogram visualization --")
    eeg = make_synthetic_eeg(n_trials=1)
    ch  = 0
    sig_orig = eeg[0, ch]

    tw = TIME_WINDOWS["T2_150-300ms"]
    fb = FREQ_BANDS_V1["gamma"]
    ablated  = stft_mask_time_freq(eeg, tw, fb)
    sig_abl  = ablated[0, ch]

    VN, VO, VNFFT = 32, 30, 128
    freqs_v, t_v, Zxx_o = scipy_stft(sig_orig, fs=FS, window="hann", nperseg=VN, noverlap=VO, nfft=VNFFT)
    _,       _,   Zxx_a = scipy_stft(sig_abl,  fs=FS, window="hann", nperseg=VN, noverlap=VO, nfft=VNFFT)
    P_orig = np.abs(Zxx_o)**2
    P_abl  = np.abs(Zxx_a)**2
    P_diff = P_orig - P_abl
    t_ms   = t_v * 1000
    tw_ms  = (tw[0]/FS*1000, tw[1]/FS*1000)

    fi = freqs_v <= 120
    fv = freqs_v[fi]
    Po = P_orig[fi]; Pa = P_abl[fi]; Pd = P_diff[fi]
    eps = 1e-12
    vm = 10*np.log10(Po.max()+eps)
    vn = vm - 60
    def db(P): return 10*np.log10(P+eps)

    # --- Fig A: before ---
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.pcolormesh(t_ms, fv, db(Po), cmap="inferno", vmin=vn, vmax=vm, shading="auto")
    ax.axvline(tw_ms[0], color="cyan",  linestyle="--", lw=1.5, label=f"Window: {tw_ms[0]:.0f}ms")
    ax.axvline(tw_ms[1], color="cyan",  linestyle="--", lw=1.5)
    ax.axhline(fb[0],    color="lime",  linestyle=":",  lw=1.5, label=f"Band: {fb[0]}-{fb[1]}Hz")
    ax.axhline(fb[1],    color="lime",  linestyle=":",  lw=1.5)
    ax.add_patch(plt.Rectangle((tw_ms[0],fb[0]),tw_ms[1]-tw_ms[0],fb[1]-fb[0],
                                fill=False,edgecolor="cyan",lw=2.5,label="Mask region"))
    plt.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Freq (Hz)"); ax.set_ylim(0,120)
    ax.set_title("STFT Spectrogram -- BEFORE masking\n(Synthetic EEG: d2Hz+a10Hz+b20Hz+g50Hz+hg90Hz)",fontweight="bold")
    ax.legend(fontsize=8,loc="upper right")
    plt.tight_layout()
    p = OUT_DIR / "verify_stft_spectrogram_before.png"
    plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # --- Fig B: after ---
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.pcolormesh(t_ms, fv, db(Pa), cmap="inferno", vmin=vn, vmax=vm, shading="auto")
    ax.axvline(tw_ms[0], color="cyan", linestyle="--", lw=1.5)
    ax.axvline(tw_ms[1], color="cyan", linestyle="--", lw=1.5)
    ax.axhline(fb[0],    color="lime", linestyle=":",  lw=1.5)
    ax.axhline(fb[1],    color="lime", linestyle=":",  lw=1.5)
    ax.add_patch(plt.Rectangle((tw_ms[0],fb[0]),tw_ms[1]-tw_ms[0],fb[1]-fb[0],
                                fill=False,edgecolor="cyan",lw=2.5))
    plt.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Freq (Hz)"); ax.set_ylim(0,120)
    ax.set_title("STFT Spectrogram -- AFTER masking (T2_150-300ms x gamma 30-80Hz)\n"
                 "Cyan box should show dark (energy removed)", fontweight="bold")
    plt.tight_layout()
    p = OUT_DIR / "verify_stft_spectrogram_after.png"
    plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # --- Fig C: 3-panel ---
    fig, axes = plt.subplots(3, 1, figsize=(12,11), sharex=True)
    im1 = axes[0].pcolormesh(t_ms,fv,db(Po),cmap="inferno",vmin=vn,vmax=vm,shading="auto")
    axes[0].add_patch(plt.Rectangle((tw_ms[0],fb[0]),tw_ms[1]-tw_ms[0],fb[1]-fb[0],fill=False,edgecolor="cyan",lw=2.5))
    plt.colorbar(im1,ax=axes[0],label="dB"); axes[0].set_ylabel("Freq (Hz)"); axes[0].set_ylim(0,120)
    axes[0].set_title("(1) BEFORE masking -- original signal", fontweight="bold")

    im2 = axes[1].pcolormesh(t_ms,fv,db(Pa),cmap="inferno",vmin=vn,vmax=vm,shading="auto")
    axes[1].add_patch(plt.Rectangle((tw_ms[0],fb[0]),tw_ms[1]-tw_ms[0],fb[1]-fb[0],fill=False,edgecolor="cyan",lw=2.5))
    plt.colorbar(im2,ax=axes[1],label="dB"); axes[1].set_ylabel("Freq (Hz)"); axes[1].set_ylim(0,120)
    axes[1].set_title("(2) AFTER masking (T2 x gamma) -- cyan box should be dark", fontweight="bold")

    dc = np.clip(Pd, 0, None)
    im3 = axes[2].pcolormesh(t_ms,fv,db(dc+eps),cmap="hot_r",vmin=vn,vmax=vm,shading="auto")
    axes[2].add_patch(plt.Rectangle((tw_ms[0],fb[0]),tw_ms[1]-tw_ms[0],fb[1]-fb[0],fill=False,edgecolor="cyan",lw=2.5))
    plt.colorbar(im3,ax=axes[2],label="dB"); axes[2].set_ylabel("Freq (Hz)"); axes[2].set_ylim(0,120)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("(3) DIFFERENCE (Before-After) -- hot region should be INSIDE cyan box", fontweight="bold")

    fig.suptitle(f"Verification T7: STFT Masking 3-Panel Compare\n"
                 f"T2_150-300ms [{tw_ms[0]:.0f}-{tw_ms[1]:.0f}ms] x gamma [{fb[0]}-{fb[1]}Hz]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    p = OUT_DIR / "verify_stft_spectrogram_compare.png"
    plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # quantitative
    tm = (t_ms >= tw_ms[0]) & (t_ms <= tw_ms[1])
    fm = (fv >= fb[0]) & (fv <= fb[1])
    e_in_o  = Po[np.ix_(fm, tm)].mean()
    e_in_a  = Pa[np.ix_(fm, tm)].mean()
    e_out_o = Po[~fm].mean()
    e_out_a = Pa[~fm].mean()
    ri = e_in_a  / (e_in_o  + 1e-10)
    ro = e_out_a / (e_out_o + 1e-10)
    check("T7a  in-box energy ratio < 0.10 (energy removed)",
          ri < 0.10, f"E_in_orig={e_in_o:.4e}, E_in_abl={e_in_a:.4e}, ratio={ri:.4f}")
    check("T7b  out-box energy ratio > 0.85 (other region intact)",
          ro > 0.85, f"E_out_orig={e_out_o:.4e}, E_out_abl={e_out_a:.4e}, ratio={ro:.4f}")


def test_t8_v2b_gamma_split():
    print("\n-- T8: v2b triple-gamma band selectivity --")
    eeg = make_synthetic_eeg()
    tw  = TIME_WINDOWS["T1_50-150ms"]
    for fb_name, fb in [("low_gamma(30-45Hz)",  FREQ_BANDS_V2["low_gamma"]),
                         ("gamma(45-70Hz)",       FREQ_BANDS_V2["gamma"]),
                         ("high_gamma(70-100Hz)", FREQ_BANDS_V2["high_gamma"])]:
        ablated  = stft_mask_time_freq(eeg, tw, fb)
        seg_orig = eeg[0, 0, tw[0]:tw[1]]
        seg_abl  = ablated[0, 0, tw[0]:tw[1]]
        f  = np.fft.rfftfreq(len(seg_orig), d=1.0/FS)
        So = np.abs(np.fft.rfft(seg_orig))**2
        Sa = np.abs(np.fft.rfft(seg_abl))**2
        m  = (f >= fb[0]) & (f <= fb[1])
        ratio = Sa[m].sum() / (So[m].sum() + 1e-10)
        check(f"T8   {fb_name} removed (ratio<0.20)",
              ratio < 0.20, f"ratio={ratio:.4f}")


def main():
    print("=" * 64)
    print(" verify_ablation_logic.py -- NeuroBridge Temporal Ablation")
    print(" Verify STFT masking logic with synthetic EEG (no model needed)")
    print("=" * 64)

    test_t1_freq_selectivity()
    test_t2_time_locality()
    test_t3_reconstruction()
    test_t4_fft_fallback()
    test_t5_mask_all_bands()
    test_t6_phase3()
    test_t7_visualization()
    test_t8_v2b_gamma_split()

    n_pass = sum(1 for s,_,_ in results if s==PASS)
    n_fail = sum(1 for s,_,_ in results if s==FAIL)

    print("\n" + "=" * 64)
    print(f"  SUMMARY: {n_pass} PASS  |  {n_fail} FAIL")
    print("=" * 64)
    if n_fail > 0:
        print("\n  FAILED tests:")
        for s, name, detail in results:
            if s == FAIL:
                print(f"    - {name}")
                if detail: print(f"        {detail}")

    report_path = OUT_DIR / "verify_ablation_logic_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("NeuroBridge Temporal Ablation -- Signal-Level Verification Report\n")
        f.write("=" * 70 + "\n\n")
        for s, name, detail in results:
            f.write(f"[{s}] {name}\n")
            if detail: f.write(f"       -> {detail}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"SUMMARY: {n_pass} PASS  |  {n_fail} FAIL\n")
        f.write("\nGenerated figures:\n")
        for fn in ["verify_stft_spectrogram_before.png",
                   "verify_stft_spectrogram_after.png",
                   "verify_stft_spectrogram_compare.png"]:
            f.write(f"  {OUT_DIR / fn}\n")

    print(f"\n  Report: {report_path}")
    print(f"  Figures: {OUT_DIR}")
    print("  Key figure: verify_stft_spectrogram_compare.png")


if __name__ == "__main__":
    main()
