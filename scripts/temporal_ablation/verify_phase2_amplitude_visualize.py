from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


def _bootstrap_site_packages() -> None:
    python_root = Path(sys.executable).resolve().parent
    candidate = python_root / "Lib" / "site-packages"
    if candidate.exists():
        site_packages = str(candidate)
        if site_packages not in sys.path:
            sys.path.append(site_packages)


_bootstrap_site_packages()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results" / "verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


amp = _load_module("ablation_temporal_amplitude_mod", SCRIPT_DIR / "ablation_temporal_amplitude.py")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def make_synthetic_eeg(n_trials: int = 6, n_channels: int = 3, n_tp: int = 250, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_tp, dtype=np.float32) / amp.FS
    base = np.zeros(n_tp, dtype=np.float32)
    components = [(6.0, 0.25), (10.0, 0.55), (20.0, 0.45), (42.0, 1.0), (68.0, 0.75)]
    for freq, magnitude in components:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        base += magnitude * np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)

    eeg = []
    for trial in range(n_trials):
        trial_channels = []
        for ch in range(n_channels):
            ch_phase = rng.uniform(0.0, 2.0 * np.pi)
            envelope = 1.0 + 0.15 * np.sin(2.0 * np.pi * 2.0 * t + ch_phase).astype(np.float32)
            noise = rng.normal(0.0, 0.03, size=n_tp).astype(np.float32)
            trial_channels.append((base * envelope + noise).astype(np.float32))
        eeg.append(trial_channels)
    return np.asarray(eeg, dtype=np.float32)


def get_tf_repr(eeg: np.ndarray, time_window: tuple[int, int], freq_band: tuple[float, float]):
    rep = amp._build_time_freq_representation(np.asarray(eeg, dtype=np.float32), time_window[0], time_window[1], freq_band[0], freq_band[1])
    coeffs = rep["coeffs"]
    freq_mask = rep["freq_mask"]
    if rep["mode"] == "fft":
        target = coeffs[:, freq_mask]
        other = coeffs[:, ~freq_mask]
    else:
        target = coeffs[:, freq_mask, :]
        other = coeffs[:, ~freq_mask, :]
    return rep, target, other


def wrapped_phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def metric_line(name: str, status: str, detail: str) -> dict[str, str]:
    print(f"[{status}] {name}: {detail}")
    return {"status": status, "name": name, "detail": detail}


def test_amplitude_scaling(results: list[dict[str, str]], figures: list[Path]) -> dict[str, float | str]:
    eeg = make_synthetic_eeg()
    tw = amp.TIME_WINDOWS["T2_150-300ms"]
    fb = amp.FREQ_BANDS["gamma"]
    alpha = 0.4
    rep_before, target_before, other_before = get_tf_repr(eeg, tw, fb)
    coeffs_mod = rep_before["coeffs"].copy()
    if rep_before["mode"] == "fft":
        coeffs_mod[:, rep_before["freq_mask"]] = rep_before["coeffs"][:, rep_before["freq_mask"]] * alpha
    else:
        coeffs_mod[:, rep_before["freq_mask"], :] = rep_before["coeffs"][:, rep_before["freq_mask"], :] * alpha
    manual = amp._invert_time_freq_representation(rep_before, coeffs_mod)
    ablated = amp.amplitude_scaling(eeg, tw, fb, alpha)
    fn_rmse = float(np.sqrt(np.mean((manual - ablated) ** 2)))

    if rep_before["mode"] == "fft":
        target_after = coeffs_mod[:, rep_before["freq_mask"]]
        other_after = coeffs_mod[:, ~rep_before["freq_mask"]]
    else:
        target_after = coeffs_mod[:, rep_before["freq_mask"], :]
        other_after = coeffs_mod[:, ~rep_before["freq_mask"], :]

    target_power_ratio = float((np.abs(target_after) ** 2).mean() / ((np.abs(target_before) ** 2).mean() + 1e-12))
    expected_power_ratio = alpha ** 2
    outside_ratio = float((np.abs(other_after) ** 2).mean() / ((np.abs(other_before) ** 2).mean() + 1e-12))
    outside_rmse = float(np.sqrt(np.mean((eeg[..., :tw[0]] - ablated[..., :tw[0]]) ** 2) + np.mean((eeg[..., tw[1]:] - ablated[..., tw[1]:]) ** 2)))

    status = PASS if abs(target_power_ratio - expected_power_ratio) < 1e-6 and abs(outside_ratio - 1.0) < 1e-6 and outside_rmse < 1e-6 and fn_rmse < 1e-7 else FAIL
    results.append(metric_line(
        "A1 scaling target-band power follows alpha^2",
        status,
        f"expected={expected_power_ratio:.3f}, actual={target_power_ratio:.3f}, outside_ratio={outside_ratio:.3f}, outside_rmse={outside_rmse:.2e}, fn_rmse={fn_rmse:.2e}",
    ))

    eeg_short = make_synthetic_eeg(seed=11)
    tw_short = amp.TIME_WINDOWS["T0_0-50ms"]
    ablated_short = amp.amplitude_scaling(eeg_short, tw_short, fb, alpha)
    _, target_short_before, _ = get_tf_repr(eeg_short, tw_short, fb)
    rep_short_after, target_short_after, _ = get_tf_repr(ablated_short, tw_short, fb)
    short_ratio = float((np.abs(target_short_after) ** 2).mean() / ((np.abs(target_short_before) ** 2).mean() + 1e-12))
    short_status = PASS if rep_short_after["mode"] == "fft" and abs(short_ratio - expected_power_ratio) < 0.08 else FAIL
    results.append(metric_line(
        "A2 short-window scaling uses FFT fallback",
        short_status,
        f"mode={rep_short_after['mode']}, expected={expected_power_ratio:.3f}, actual={short_ratio:.3f}",
    ))

    figures.extend(plot_stft_triplet(
        eeg[0, 0],
        ablated[0, 0],
        tw,
        fb,
        "ablation_temporal_amplitude_scaling",
        "Amplitude Scaling",
    ))
    return {"target_power_ratio": target_power_ratio, "short_window_mode": str(rep_short_after["mode"])}


def test_phase_randomization(results: list[dict[str, str]], figures: list[Path]) -> dict[str, float]:
    eeg = make_synthetic_eeg(seed=17)
    tw = amp.TIME_WINDOWS["T2_150-300ms"]
    fb = amp.FREQ_BANDS["gamma"]

    sample = np.exp(1j * np.linspace(0.0, np.pi, 128, dtype=np.float32)).reshape(8, 16)
    randomized = amp._randomize_phase_values(sample, 0.5, np.random.default_rng(23))
    changed_fraction = float((np.abs(wrapped_phase_diff(np.angle(sample), np.angle(randomized))) > 1e-3).mean())
    amplitude_ratio = float(np.abs(randomized).mean() / (np.abs(sample).mean() + 1e-12))
    sample_status = PASS if 0.35 <= changed_fraction <= 0.65 and abs(amplitude_ratio - 1.0) < 1e-6 else FAIL
    results.append(metric_line(
        "B1 phase randomization matches rand_ratio semantics",
        sample_status,
        f"changed_fraction={changed_fraction:.3f}, amplitude_ratio={amplitude_ratio:.6f}",
    ))

    identity = amp.phase_randomization(eeg, tw, fb, 0.0, np.random.default_rng(31))
    identity_rmse = float(np.sqrt(np.mean((identity - eeg) ** 2)))
    identity_status = PASS if identity_rmse < 1e-6 else FAIL
    results.append(metric_line(
        "B2 phase rand_ratio=0 keeps signal unchanged",
        identity_status,
        f"rmse={identity_rmse:.2e}",
    ))

    rep_before, target_before, other_before = get_tf_repr(eeg, tw, fb)
    coeffs_mod = rep_before["coeffs"].copy()
    if rep_before["mode"] == "fft":
        target_mod = amp._randomize_phase_values(rep_before["coeffs"][:, rep_before["freq_mask"]], 1.0, np.random.default_rng(5))
        coeffs_mod[:, rep_before["freq_mask"]] = target_mod
    else:
        target_mod = amp._randomize_phase_values(rep_before["coeffs"][:, rep_before["freq_mask"], :], 1.0, np.random.default_rng(5))
        coeffs_mod[:, rep_before["freq_mask"], :] = target_mod
    manual = amp._invert_time_freq_representation(rep_before, coeffs_mod)
    fully_random = amp.phase_randomization(eeg, tw, fb, 1.0, np.random.default_rng(5))
    fn_rmse = float(np.sqrt(np.mean((manual - fully_random) ** 2)))

    target_after = target_mod
    if rep_before["mode"] == "fft":
        other_after = coeffs_mod[:, ~rep_before["freq_mask"]]
    else:
        other_after = coeffs_mod[:, ~rep_before["freq_mask"], :]
    mag_ratio = float(np.abs(target_after).mean() / (np.abs(target_before).mean() + 1e-12))
    mean_phase_shift = float(np.mean(np.abs(wrapped_phase_diff(np.angle(target_after), np.angle(target_before)))))
    outside_ratio = float((np.abs(other_after) ** 2).mean() / ((np.abs(other_before) ** 2).mean() + 1e-12))
    signal_status = PASS if abs(mag_ratio - 1.0) < 1e-6 and mean_phase_shift > 0.7 and abs(outside_ratio - 1.0) < 1e-6 and fn_rmse < 1e-7 else FAIL
    results.append(metric_line(
        "B3 phase randomization preserves magnitude and rotates target phase",
        signal_status,
        f"mag_ratio={mag_ratio:.3f}, mean_phase_shift={mean_phase_shift:.3f}rad, outside_ratio={outside_ratio:.3f}, fn_rmse={fn_rmse:.2e}",
    ))

    figures.extend(plot_stft_triplet(
        eeg[0, 0],
        fully_random[0, 0],
        tw,
        fb,
        "ablation_temporal_amplitude_phase_randomization",
        "Phase Randomization",
    ))
    return {"phase_changed_fraction": changed_fraction, "phase_mag_ratio": mag_ratio}


def test_gaussian_noise(results: list[dict[str, str]], figures: list[Path]) -> dict[str, float]:
    eeg = make_synthetic_eeg(seed=29)
    tw = amp.TIME_WINDOWS["T2_150-300ms"]
    fb = amp.FREQ_BANDS["gamma"]
    snr_db = 0.0
    rep_before, target_before, other_before = get_tf_repr(eeg, tw, fb)
    coeffs_mod = rep_before["coeffs"].copy()
    if rep_before["mode"] == "fft":
        target_view = rep_before["coeffs"][:, rep_before["freq_mask"]]
    else:
        target_view = rep_before["coeffs"][:, rep_before["freq_mask"], :]
    signal_power = float((np.abs(target_view) ** 2).mean())
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)
    rng = np.random.default_rng(41)
    noise = rng.normal(0.0, noise_std, size=target_view.shape) + 1j * rng.normal(0.0, noise_std, size=target_view.shape)
    if rep_before["mode"] == "fft":
        coeffs_mod[:, rep_before["freq_mask"]] = target_view + noise.astype(np.complex64)
        other_after = coeffs_mod[:, ~rep_before["freq_mask"]]
    else:
        coeffs_mod[:, rep_before["freq_mask"], :] = target_view + noise.astype(np.complex64)
        other_after = coeffs_mod[:, ~rep_before["freq_mask"], :]
    manual = amp._invert_time_freq_representation(rep_before, coeffs_mod)
    noisy = amp.gaussian_noise_injection(eeg, tw, fb, snr_db, np.random.default_rng(41))
    fn_rmse = float(np.sqrt(np.mean((manual - noisy) ** 2)))

    target_after = target_view + noise
    noise = target_after - target_before
    signal_power = float((np.abs(target_before) ** 2).mean())
    noise_power = float((np.abs(noise) ** 2).mean())
    achieved_snr_db = 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))
    outside_ratio = float((np.abs(other_after) ** 2).mean() / ((np.abs(other_before) ** 2).mean() + 1e-12))
    status = PASS if abs(achieved_snr_db - snr_db) < 0.2 and abs(outside_ratio - 1.0) < 1e-6 and fn_rmse < 1e-7 else FAIL
    results.append(metric_line(
        "C1 gaussian noise hits requested target-region SNR",
        status,
        f"target_snr={snr_db:.1f}dB, achieved_snr={achieved_snr_db:.2f}dB, outside_ratio={outside_ratio:.3f}, fn_rmse={fn_rmse:.2e}",
    ))

    figures.extend(plot_stft_triplet(
        eeg[0, 0],
        noisy[0, 0],
        tw,
        fb,
        "ablation_temporal_amplitude_gaussian_noise",
        "Gaussian Noise Injection",
    ))
    return {"achieved_snr_db": achieved_snr_db}


def plot_stft_triplet(
    before: np.ndarray,
    after: np.ndarray,
    time_window: tuple[int, int],
    freq_band: tuple[float, float],
    prefix: str,
    title: str,
) -> list[Path]:
    vn = 32
    vo = 16
    vf = 128
    freqs, times, z_before = stft(before, fs=amp.FS, window="hann", nperseg=vn, noverlap=vo, nfft=vf)
    _, _, z_after = stft(after, fs=amp.FS, window="hann", nperseg=vn, noverlap=vo, nfft=vf)
    power_before = np.abs(z_before) ** 2
    power_after = np.abs(z_after) ** 2
    power_diff = np.abs(power_after - power_before)
    time_ms = times * 1000.0
    mask_freq = freqs <= 120.0
    freqs = freqs[mask_freq]
    power_before = power_before[mask_freq]
    power_after = power_after[mask_freq]
    power_diff = power_diff[mask_freq]
    eps = 1e-12
    vmax = 10.0 * np.log10(power_before.max() + eps)
    vmin = vmax - 60.0
    tw_ms = (time_window[0] / amp.FS * 1000.0, time_window[1] / amp.FS * 1000.0)

    def _plot(power: np.ndarray, file_name: str, subtitle: str, cmap: str) -> Path:
        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(time_ms, freqs, 10.0 * np.log10(power + eps), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.axvline(tw_ms[0], color="cyan", linestyle="--", linewidth=1.5)
        ax.axvline(tw_ms[1], color="cyan", linestyle="--", linewidth=1.5)
        ax.axhline(freq_band[0], color="lime", linestyle=":", linewidth=1.5)
        ax.axhline(freq_band[1], color="lime", linestyle=":", linewidth=1.5)
        ax.add_patch(plt.Rectangle((tw_ms[0], freq_band[0]), tw_ms[1] - tw_ms[0], freq_band[1] - freq_band[0], fill=False, edgecolor="cyan", linewidth=2.5))
        plt.colorbar(mesh, ax=ax, label="Power (dB)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, 120)
        ax.set_title(f"{prefix}.py — {subtitle}\n{title} @ {freq_band[0]}-{freq_band[1]}Hz / {tw_ms[0]:.0f}-{tw_ms[1]:.0f}ms", fontweight="bold")
        plt.tight_layout()
        out_path = OUT_DIR / file_name
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return out_path

    return [
        _plot(power_before, f"{prefix}_stft_before.png", "STFT BEFORE", "inferno"),
        _plot(power_after, f"{prefix}_stft_after.png", "STFT AFTER", "inferno"),
        _plot(power_diff, f"{prefix}_stft_diff.png", "STFT DIFF |after-before|", "magma"),
    ]


def write_visualize_fixtures() -> tuple[Path, Path]:
    phase1_path = OUT_DIR / "visualize_fixture_phase1.csv"
    phase2_path = OUT_DIR / "visualize_fixture_phase2.csv"
    phase1_rows = [
        {"name": "baseline", "time_window": "", "freq_band": "", "priority": "control", "top1": "0.4200", "top5": "0.7600", "mean_rank": "4.2", "median_rank": "3", "top1_drop": "", "top5_drop": ""},
        {"name": "T1_50-150ms__gamma", "time_window": "T1_50-150ms", "freq_band": "gamma", "priority": "5", "top1": "0.3100", "top5": "0.6900", "mean_rank": "6.1", "median_rank": "5", "top1_drop": "0.1100", "top5_drop": "0.0700"},
        {"name": "T2_150-300ms__gamma", "time_window": "T2_150-300ms", "freq_band": "gamma", "priority": "5", "top1": "0.2800", "top5": "0.6500", "mean_rank": "6.8", "median_rank": "6", "top1_drop": "0.1400", "top5_drop": "0.1100"},
        {"name": "T2_150-300ms__beta", "time_window": "T2_150-300ms", "freq_band": "beta", "priority": "4", "top1": "0.3300", "top5": "0.7100", "mean_rank": "5.4", "median_rank": "4", "top1_drop": "0.0900", "top5_drop": "0.0500"},
    ]
    phase2_rows = [
        {"condition": "T2_150-300ms__gamma", "perturbation": "scaling", "param_name": "alpha", "param_value": "0.0", "top1": "0.2400", "top5": "0.6000", "mean_rank": "7.4", "median_rank": "6", "top1_drop": "0.1800", "top5_drop": "0.1600"},
        {"condition": "T2_150-300ms__gamma", "perturbation": "scaling", "param_name": "alpha", "param_value": "1.0", "top1": "0.4200", "top5": "0.7600", "mean_rank": "4.2", "median_rank": "3", "top1_drop": "0.0000", "top5_drop": "0.0000"},
        {"condition": "T2_150-300ms__gamma", "perturbation": "phase_rand", "param_name": "rand_ratio", "param_value": "0.0", "top1": "0.4200", "top5": "0.7600", "mean_rank": "4.2", "median_rank": "3", "top1_drop": "0.0000", "top5_drop": "0.0000"},
        {"condition": "T2_150-300ms__gamma", "perturbation": "phase_rand", "param_name": "rand_ratio", "param_value": "1.0", "top1": "0.3000", "top5": "0.6800", "mean_rank": "6.0", "median_rank": "5", "top1_drop": "0.1200", "top5_drop": "0.0800"},
        {"condition": "T2_150-300ms__gamma", "perturbation": "gaussian_noise", "param_name": "snr_db", "param_value": "20.0", "top1": "0.4000", "top5": "0.7500", "mean_rank": "4.4", "median_rank": "4", "top1_drop": "0.0200", "top5_drop": "0.0100"},
        {"condition": "T2_150-300ms__gamma", "perturbation": "gaussian_noise", "param_name": "snr_db", "param_value": "-10.0", "top1": "0.2500", "top5": "0.6200", "mean_rank": "7.1", "median_rank": "6", "top1_drop": "0.1700", "top5_drop": "0.1400"},
        {"condition": "T2_150-300ms__beta", "perturbation": "scaling", "param_name": "alpha", "param_value": "0.0", "top1": "0.2900", "top5": "0.6800", "mean_rank": "6.3", "median_rank": "5", "top1_drop": "0.1300", "top5_drop": "0.0800"},
        {"condition": "T2_150-300ms__beta", "perturbation": "scaling", "param_name": "alpha", "param_value": "1.0", "top1": "0.4200", "top5": "0.7600", "mean_rank": "4.2", "median_rank": "3", "top1_drop": "0.0000", "top5_drop": "0.0000"},
        {"condition": "T2_150-300ms__beta", "perturbation": "phase_rand", "param_name": "rand_ratio", "param_value": "0.0", "top1": "0.4200", "top5": "0.7600", "mean_rank": "4.2", "median_rank": "3", "top1_drop": "0.0000", "top5_drop": "0.0000"},
        {"condition": "T2_150-300ms__beta", "perturbation": "phase_rand", "param_name": "rand_ratio", "param_value": "1.0", "top1": "0.3400", "top5": "0.7000", "mean_rank": "5.6", "median_rank": "5", "top1_drop": "0.0800", "top5_drop": "0.0600"},
        {"condition": "T2_150-300ms__beta", "perturbation": "gaussian_noise", "param_name": "snr_db", "param_value": "20.0", "top1": "0.4100", "top5": "0.7500", "mean_rank": "4.3", "median_rank": "4", "top1_drop": "0.0100", "top5_drop": "0.0100"},
        {"condition": "T2_150-300ms__beta", "perturbation": "gaussian_noise", "param_name": "snr_db", "param_value": "-10.0", "top1": "0.3100", "top5": "0.6600", "mean_rank": "6.2", "median_rank": "5", "top1_drop": "0.1100", "top5_drop": "0.1000"},
    ]
    with phase1_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase1_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase1_rows)
    with phase2_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase2_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase2_rows)
    return phase1_path, phase2_path


def test_visualize_chain(results: list[dict[str, str]], figures: list[Path]) -> dict[str, object]:
    phase1_csv, phase2_csv = write_visualize_fixtures()
    output_dir = OUT_DIR / "ablation_visualize_outputs"
    report_path = OUT_DIR / "ablation_visualize_chain_report.json"
    command = [
        sys.executable,
        "-S",
        str(SCRIPT_DIR / "ablation_visualize.py"),
        "--phase",
        "12",
        "--phase1-csv",
        str(phase1_csv),
        "--phase2-csv",
        str(phase2_csv),
        "--output-dir",
        str(output_dir),
        "--report-path",
        str(report_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        results.append(metric_line("V1 ablation_visualize execution", FAIL, completed.stderr.strip() or completed.stdout.strip()))
        return {"returncode": completed.returncode, "generated": []}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_names = {
        "fig1_stft_heatmap.png",
        "fig2_phase1_bar.png",
        "fig3_amplitude_scaling.png",
        "fig4_phase_randomization.png",
        "fig5_noise_injection.png",
        "fig6_phase2_summary.png",
    }
    generated = {Path(p).name for p in report.get("generated_figures", [])}
    missing = sorted(expected_names - generated)
    missing_fields = report["phase1"]["missing_fields"] + report["phase2"]["missing_fields"]
    status = PASS if not missing and not missing_fields else FAIL
    results.append(metric_line(
        "V1 ablation_visualize input/output field chain",
        status,
        f"missing_fields={missing_fields or '[]'}, missing_figures={missing or '[]'}",
    ))
    for item in report.get("generated_figures", []):
        figures.append(Path(item))
    return report


def write_reports(metrics: list[dict[str, str]], figures: list[Path], sections: dict[str, object]) -> tuple[Path, Path]:
    summary_path = OUT_DIR / "phase2_amplitude_validation_report.txt"
    html_path = OUT_DIR / "phase2_amplitude_validation_report.html"
    pass_count = sum(1 for item in metrics if item["status"] == PASS)
    fail_count = sum(1 for item in metrics if item["status"] == FAIL)
    warn_count = sum(1 for item in metrics if item["status"] == WARN)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Phase 2 Amplitude Perturbation & Visualization Validation Report\n")
        f.write("=" * 72 + "\n\n")
        for item in metrics:
            f.write(f"[{item['status']}] {item['name']}\n")
            f.write(f"  -> {item['detail']}\n")
        f.write("\nSummary\n")
        f.write(f"PASS={pass_count} FAIL={fail_count} WARN={warn_count}\n\n")
        f.write("Key metrics\n")
        for key, value in sections.items():
            f.write(f"- {key}: {json.dumps(value, ensure_ascii=False)}\n")
        f.write("\nGenerated figures\n")
        for fig in figures:
            f.write(f"- {fig}\n")

    rows_html = "\n".join(
        f"<tr><td>{item['status']}</td><td>{item['name']}</td><td><code>{item['detail']}</code></td></tr>"
        for item in metrics
    )
    figure_html = "\n".join(f"<li><code>{fig}</code></li>" for fig in figures)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Phase 2 Temporal Ablation Validation</title>
<style>
body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; vertical-align: top; }}
th {{ background: #1d4ed8; color: white; }}
.PASS {{ color: #15803d; font-weight: 700; }}
.FAIL {{ color: #b91c1c; font-weight: 700; }}
.WARN {{ color: #b45309; font-weight: 700; }}
code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Phase 2 振幅扰动与可视化链路验证</h1>
<p>脚本目录：<code>{SCRIPT_DIR}</code></p>
<p>结论：PASS={pass_count} / FAIL={fail_count} / WARN={warn_count}</p>
<table>
<tr><th>状态</th><th>检查项</th><th>细节</th></tr>
{rows_html}
</table>
<h2>关键指标</h2>
<pre>{json.dumps(sections, ensure_ascii=False, indent=2)}</pre>
<h2>生成图片</h2>
<ul>
{figure_html}
</ul>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return summary_path, html_path


def main() -> None:
    metrics: list[dict[str, str]] = []
    figures: list[Path] = []
    sections = {
        "amplitude_scaling": test_amplitude_scaling(metrics, figures),
        "phase_randomization": test_phase_randomization(metrics, figures),
        "gaussian_noise": test_gaussian_noise(metrics, figures),
        "ablation_visualize": test_visualize_chain(metrics, figures),
    }
    summary_path, html_path = write_reports(metrics, figures, sections)
    print(f"\nReport TXT: {summary_path}")
    print(f"Report HTML: {html_path}")
    print(f"Figures: {OUT_DIR}")


if __name__ == "__main__":
    main()
