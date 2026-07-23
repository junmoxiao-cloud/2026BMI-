"""
ablation_visualize.py — Phase 1 & 2 Result Visualization
=========================================================
读取 Phase 1 (STFT) 和 Phase 2 (振幅扰动) 的实验结果，生成以下图表：

  Phase 1:
    Fig 1. 时间窗 x 频段的 Top-1 Accuracy Drop 热力图
    Fig 2. 各条件 Top-1/Top-5 柱状图（按 drop 排序）
  Phase 2:
    Fig 3. 振幅缩放曲线（准确率 vs alpha）
    Fig 4. 相位随机化曲线（准确率 vs rand_ratio）
    Fig 5. 高斯噪声曲线（准确率 vs SNR_dB）
    Fig 6. Phase 2 三种扰动综合对比

用法：
  python ablation_visualize.py
  python ablation_visualize.py --phase 1
  python ablation_visualize.py --phase 2

作者：乔钰成 / NEOschool 项目组
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_CSV = SCRIPT_DIR / "results" / "temporal_stft_ablation" / "stft_ablation_results.csv"
PHASE2_CSV = SCRIPT_DIR / "results" / "temporal_amplitude_ablation" / "amplitude_ablation_results.csv"
OUTPUT_DIR = SCRIPT_DIR / "results" / "figures"

TW_ORDER = ["T0_0-50ms","T1_50-150ms","T2_150-300ms","T3_300-500ms","T4_500-800ms"]
FB_ORDER = ["delta","theta","alpha","beta","gamma","hi_gamma"]
TW_LABELS = {
    "T0_0-50ms":    "0-52ms\n(V1 onset)",
    "T1_50-150ms":  "52-152ms\n(V1 peak 101ms)",
    "T2_150-300ms": "152-300ms\n(N170/Thorpe 152ms)",
    "T3_300-500ms": "300-500ms\n(P300/Theta)",
    "T4_500-800ms": "500-800ms\n(Late LPC)",
}
FB_LABELS = {
    "delta":"delta\n1-4Hz","theta":"theta\n4-8Hz","alpha":"alpha\n8-13Hz",
    "beta":"beta\n13-30Hz","gamma":"gamma\n30-80Hz","hi_gamma":"hi-gamma\n80-120Hz",
}
COLORS = ["#E74C3C","#E67E22","#3498DB","#2ECC71","#9B59B6","#1ABC9C"]

def load_phase1(path):
    if not path.exists(): return {}, None
    data, baseline = {}, None
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["name"] == "baseline":
                baseline = float(r["top1"]); continue
            tw, fb = r.get("time_window",""), r.get("freq_band","")
            d = r.get("top1_drop","")
            if tw and fb and d not in ("","None",None):
                try: data[(tw,fb)] = float(d)
                except: pass
    return data, baseline

def load_csv_rows(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def plot_heatmap(data, baseline_top1, out_dir):
    matrix = np.full((len(TW_ORDER), len(FB_ORDER)), np.nan)
    for i,tw in enumerate(TW_ORDER):
        for j,fb in enumerate(FB_ORDER):
            if (tw,fb) in data: matrix[i,j] = data[(tw,fb)]
    fig, ax = plt.subplots(figsize=(11,6))
    vmax = max(0.15, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 0.15
    vmin = min(0.0,  np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0.0
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    for i in range(len(TW_ORDER)):
        for j in range(len(FB_ORDER)):
            v = matrix[i,j]
            if not np.isnan(v):
                c = "white" if abs(v) > vmax*0.6 else "black"
                ax.text(j,i,f"{v:+.3f}",ha="center",va="center",fontsize=9,color=c,fontweight="bold")
            else:
                ax.text(j,i,"N/A",ha="center",va="center",fontsize=8,color="gray")
    top_val = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0
    for i in range(len(TW_ORDER)):
        for j in range(len(FB_ORDER)):
            if not np.isnan(matrix[i,j]) and matrix[i,j] >= top_val*0.9:
                ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="gold",linewidth=3))
    ax.set_xticks(range(len(FB_ORDER)))
    ax.set_xticklabels([FB_LABELS[fb] for fb in FB_ORDER], fontsize=9)
    ax.set_yticks(range(len(TW_ORDER)))
    ax.set_yticklabels([TW_LABELS[tw] for tw in TW_ORDER], fontsize=9)
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Top-1 Accuracy Drop  (higher = more important)", fontsize=10)
    bl = f"  [Baseline Top-1: {baseline_top1:.4f}]" if baseline_top1 else ""
    ax.set_title(f"Phase 1: STFT Temporal Ablation — Top-1 Accuracy Drop Heatmap{bl}\n"
                 "(NeuroBridge sub-08 | THINGS-EEG2 | Cichy 2014 / Thorpe 1996 time windows)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Frequency Band  [Fries 2015: gamma=feedforward, beta=feedback]", fontsize=9)
    ax.set_ylabel("Time Window", fontsize=10)
    plt.tight_layout()
    out = out_dir / "fig1_stft_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def plot_phase1_bar(rows, out_dir):
    valid = [r for r in rows
             if r.get("name") not in ("baseline","full_mask_all","random_control","")
             and r.get("top1_drop") not in ("","None",None)]
    if not valid: print("  [Skip Fig2] no data"); return
    valid.sort(key=lambda x: float(x["top1_drop"]), reverse=True)
    names = [r["name"].replace("__","\n") for r in valid]
    d1 = [float(r["top1_drop"]) for r in valid]
    d5 = [float(r.get("top5_drop") or 0) for r in valid]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(12,len(names)*1.2), 6))
    ax.bar(x-w/2, d1, w, label="Top-1 Drop", color="#E74C3C", alpha=0.85)
    ax.bar(x+w/2, d5, w, label="Top-5 Drop", color="#3498DB", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Accuracy Drop", fontsize=10)
    ax.set_title("Phase 1: STFT Ablation — Accuracy Drop per Condition (sorted)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig2_phase1_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def plot_scaling_curve(rows, out_dir):
    data = [r for r in rows if r.get("perturbation")=="scaling"]
    if not data: print("  [Skip Fig3] no scaling data"); return
    conds = sorted(set(r["condition"] for r in data))
    fig, ax = plt.subplots(figsize=(9,5))
    for i,cond in enumerate(conds):
        sub = sorted([r for r in data if r["condition"]==cond], key=lambda x: float(x["param_value"]))
        ax.plot([float(r["param_value"]) for r in sub],
                [float(r["top1"]) for r in sub],
                "o-", color=COLORS[i%len(COLORS)], linewidth=2, markersize=6,
                label=cond.replace("__"," / "))
    baseline_vals = [float(r["top1"]) for r in data if float(r["param_value"])==1.0]
    if baseline_vals:
        ax.axhline(np.mean(baseline_vals), color="gray", linestyle="--", linewidth=1.5, label="Baseline (alpha=1.0)")
    ax.axvspan(0.6, 0.8, alpha=0.10, color="orange", zorder=0, label="AD range (alpha=0.6-0.8)")
    ax.set_xlabel("Amplitude Scale Factor (alpha)  [0=zeroed, 1=original, >1=enhanced]", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy", fontsize=10)
    ax.set_title("Phase 2A: Amplitude Scaling\n"
                 "(orange = AD amplitude reduction range; Haufe 2014 interpretation framework)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 2.1)
    plt.tight_layout()
    out = out_dir / "fig3_amplitude_scaling.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def plot_phase_curve(rows, out_dir):
    data = [r for r in rows if r.get("perturbation")=="phase_rand"]
    if not data: print("  [Skip Fig4] no phase data"); return
    conds = sorted(set(r["condition"] for r in data))
    fig, ax = plt.subplots(figsize=(8,5))
    for i,cond in enumerate(conds):
        sub = sorted([r for r in data if r["condition"]==cond], key=lambda x: float(x["param_value"]))
        ax.plot([float(r["param_value"]) for r in sub],
                [float(r["top1"]) for r in sub],
                "s-", color=COLORS[i%len(COLORS)], linewidth=2, markersize=7,
                label=cond.replace("__"," / "))
    ax.set_xlabel("Phase Randomization Ratio  (0=original phase, 1=fully random phase)", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy", fontsize=10)
    ax.set_title("Phase 2B: Phase Randomization — Power preserved, temporal structure destroyed\n"
                 "(if drop >> Scaling drop: model uses phase; if similar: model uses power spectrum)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    out = out_dir / "fig4_phase_randomization.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def plot_noise_curve(rows, out_dir):
    data = [r for r in rows if r.get("perturbation")=="gaussian_noise"]
    if not data: print("  [Skip Fig5] no noise data"); return
    conds = sorted(set(r["condition"] for r in data))
    fig, ax = plt.subplots(figsize=(8,5))
    for i,cond in enumerate(conds):
        sub = sorted([r for r in data if r["condition"]==cond],
                     key=lambda x: float(x["param_value"]), reverse=True)
        ax.plot([float(r["param_value"]) for r in sub],
                [float(r["top1"]) for r in sub],
                "^-", color=COLORS[i%len(COLORS)], linewidth=2, markersize=7,
                label=cond.replace("__"," / "))
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2, label="SNR=0dB")
    ax.set_xlabel("SNR (dB)  [right=cleaner signal, left=noisier]", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy", fontsize=10)
    ax.set_title("Phase 2C: Gaussian Noise Injection — BCI Robustness Test\n"
                 "(find critical SNR where accuracy collapses)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig5_noise_injection.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def plot_phase2_summary(rows, out_dir):
    conds = sorted(set(r["condition"] for r in rows))
    if not conds: return
    pert_info = [
        ("scaling",        0.0,   "Amplitude Scaling\n(alpha=0.0)"),
        ("phase_rand",     1.0,   "Phase Randomization\n(ratio=1.0)"),
        ("gaussian_noise", -10.0, "Gaussian Noise\n(SNR=-10dB)"),
    ]
    x = np.arange(len(conds)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(conds)*2.5), 5))
    for pi,(pt,ep,pl) in enumerate(pert_info):
        drops = []
        for cond in conds:
            m = [r for r in rows if r["condition"]==cond and r["perturbation"]==pt
                 and abs(float(r["param_value"])-ep)<0.01]
            drops.append(float(m[0]["top1_drop"]) if m else 0.0)
        ax.bar(x+(pi-1)*w, drops, w, label=pl, color=COLORS[pi], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("__","\n") for c in conds], fontsize=9)
    ax.set_ylabel("Top-1 Accuracy Drop  (higher = more important)", fontsize=10)
    ax.set_title("Phase 2 Summary: Three Perturbation Types at Maximum Strength\n"
                 "(compare: Scaling vs Phase-rand vs Noise across conditions)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig6_phase2_summary.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualization for Phase 1 & 2 Ablation")
    parser.add_argument("--phase", type=int, choices=[1,2,12], default=12,
                        help="要可视化的阶段 (1 / 2 / 12=两者都画)")
    parser.add_argument("--phase1-csv", type=Path, default=PHASE1_CSV)
    parser.add_argument("--phase2-csv", type=Path, default=PHASE2_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Visualize] Output dir: {args.output_dir}")

    if args.phase in (1, 12):
        print("\n--- Phase 1 Figures ---")
        data, baseline_top1 = load_phase1(args.phase1_csv)
        rows_p1 = load_csv_rows(args.phase1_csv)
        if data:
            plot_heatmap(data, baseline_top1, args.output_dir)
        else:
            print("  [Skip Fig1] Phase 1 CSV not found or empty")
        if rows_p1:
            plot_phase1_bar(rows_p1, args.output_dir)

    if args.phase in (2, 12):
        print("\n--- Phase 2 Figures ---")
        rows_p2 = load_csv_rows(args.phase2_csv)
        if rows_p2:
            plot_scaling_curve(rows_p2, args.output_dir)
            plot_phase_curve(rows_p2, args.output_dir)
            plot_noise_curve(rows_p2, args.output_dir)
            plot_phase2_summary(rows_p2, args.output_dir)
        else:
            print("  [Skip Phase2] Phase 2 CSV not found or empty")

    print(f"\n✅  All figures saved to: {args.output_dir}")
    print("    fig1_stft_heatmap.png        — Phase 1 时频热力图")
    print("    fig2_phase1_bar.png          — Phase 1 柱状图")
    print("    fig3_amplitude_scaling.png   — Phase 2A 振幅缩放曲线")
    print("    fig4_phase_randomization.png — Phase 2B 相位随机化曲线")
    print("    fig5_noise_injection.png     — Phase 2C 噪声注入曲线")
    print("    fig6_phase2_summary.png      — Phase 2 三种扰动综合对比")

if __name__ == "__main__":
    main()
