import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "assets" / "temporal_ablation"

PHASE1_CSV = RESULTS_DIR / "temporal_stft_ablation" / "stft_ablation_results.csv"
PHASE2_CSV = RESULTS_DIR / "temporal_amplitude_ablation" / "amplitude_ablation_results.csv"
PHASE3_CSV = RESULTS_DIR / "phase3_full_freq_masking" / "phase3_window_masking_results.csv"
MCNEMAR_CSV = RESULTS_DIR / "temporal_stft_ablation" / "mcnemar_fdr_results.csv"

TW_ORDER = ["T0_0-50ms", "T1_50-150ms", "T2_150-300ms", "T3_300-500ms", "T4_500-800ms"]
FB_ORDER = ["delta", "theta", "alpha", "beta", "low_gamma", "gamma", "high_gamma"]

TW_LABELS = {
    "T0_0-50ms":    "0-52ms\n(V1 onset)",
    "T1_50-150ms":  "52-152ms\n(V1 peak)",
    "T2_150-300ms": "152-300ms\n(N170/Thorpe)",
    "T3_300-500ms": "300-500ms\n(P300/Theta)",
    "T4_500-800ms": "500-800ms\n(Late LPC)",
}

FB_LABELS = {
    "delta":"delta\n1-4Hz", "theta":"theta\n4-8Hz", "alpha":"alpha\n8-13Hz",
    "beta":"beta\n13-30Hz", "low_gamma":"low_gamma\n30-45Hz",
    "gamma":"gamma\n45-70Hz", "high_gamma":"high_gamma\n70-100Hz",
}

COLORS = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71", "#9B59B6", "#1ABC9C"]

def load_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_mcnemar_fdr(path):
    if not path or not path.exists(): return {}
    sig_map = {}
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cond = r.get("Condition", "")
            sig = r.get("Significant_FDR_0.05", "False")
            if cond and sig == "True":
                sig_map[cond] = True
    return sig_map

def load_phase1_data(rows):
    data = {}
    baseline = None
    for r in rows:
        if r.get("name") == "baseline":
            baseline = float(r["top1"])
            continue
        tw, fb = r.get("time_window",""), r.get("freq_band","")
        d = r.get("top1_drop","")
        if tw and fb and d not in ("","None",None):
            try: data[(tw,fb)] = float(d)
            except: pass
    return data, baseline

def plot_panel_A_phase3(ax, rows):
    """Panel A: Phase 3 结果（使用 full_freq 数据，画柱状图）"""
    valid = [r for r in rows if r.get("time_window") in TW_ORDER and r.get("top1_drop") not in ("","None",None)]
    if not valid:
        ax.text(0.5, 0.5, "No Phase 3 Data", ha="center", va="center")
        ax.set_title("A. Phase 3: Full-Frequency Window Masking")
        return
    
    # 按照 TW_ORDER 排序
    valid.sort(key=lambda x: TW_ORDER.index(x["time_window"]) if x["time_window"] in TW_ORDER else 99)
    names = [TW_LABELS.get(r["time_window"], r["time_window"]).replace("\n", " ") for r in valid]
    d1 = [float(r["top1_drop"]) for r in valid]
    
    x = np.arange(len(names))
    ax.bar(x, d1, color="#8E44AD", alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    
    for i, v in enumerate(d1):
        ax.text(i, v + 0.005 if v >= 0 else v - 0.015, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Top-1 Accuracy Drop", fontsize=10)
    ax.set_title("A. Phase 3: Full-Frequency Window Masking", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

def plot_panel_B_heatmap(ax, data, baseline_top1, sig_map):
    """Panel B: Phase 1 时频热力图"""
    matrix = np.full((len(TW_ORDER), len(FB_ORDER)), np.nan)
    for i, tw in enumerate(TW_ORDER):
        for j, fb in enumerate(FB_ORDER):
            if (tw, fb) in data:
                matrix[i, j] = data[(tw, fb)]
                
    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No Phase 1 Data", ha="center", va="center")
        ax.set_title("B. Phase 1: STFT Ablation Heatmap")
        return

    vmax = max(0.15, np.nanmax(matrix))
    vmin = min(0.0,  np.nanmin(matrix))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    
    for i in range(len(TW_ORDER)):
        for j in range(len(FB_ORDER)):
            v = matrix[i, j]
            if not np.isnan(v):
                c = "white" if abs(v) > vmax*0.6 else "black"
                text_str = f"{v:+.3f}"
                cond_name = f"{TW_ORDER[i]}__{FB_ORDER[j]}"
                if sig_map.get(cond_name):
                    text_str += "\n*"
                ax.text(j, i, text_str, ha="center", va="center", fontsize=8, color=c, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="gray")
                
    top_val = np.nanmax(matrix)
    for i in range(len(TW_ORDER)):
        for j in range(len(FB_ORDER)):
            if not np.isnan(matrix[i,j]) and matrix[i,j] >= top_val*0.9:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="gold", linewidth=2))
                
    ax.set_xticks(range(len(FB_ORDER)))
    ax.set_xticklabels([FB_LABELS[fb].replace("\n", " ") for fb in FB_ORDER], fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(TW_ORDER)))
    ax.set_yticklabels([TW_LABELS[tw].replace("\n", " ") for tw in TW_ORDER], fontsize=9)
    
    cb = plt.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Top-1 Acc Drop", fontsize=9)
    
    bl = f" [Baseline: {baseline_top1:.4f}]" if baseline_top1 else ""
    ax.set_title(f"B. Phase 1: STFT Top-1 Drop Heatmap{bl}", fontsize=12, fontweight="bold")

def plot_panel_C_top10(ax, rows):
    """Panel C: Phase 1 Top-10 柱状图"""
    valid = [r for r in rows
             if r.get("name") not in ("baseline","full_mask_all","random_control","")
             and r.get("top1_drop") not in ("","None",None) and "__" in r.get("name", "")]
    if not valid:
        ax.text(0.5, 0.5, "No Phase 1 Data", ha="center", va="center")
        ax.set_title("C. Phase 1: Top-10 Accuracy Drop")
        return
        
    valid.sort(key=lambda x: float(x["top1_drop"]), reverse=True)
    valid = valid[:10]
    
    names = [r["name"].replace("__", "\n") for r in valid]
    d1 = [float(r["top1_drop"]) for r in valid]
    d5 = [float(r.get("top5_drop") or 0) for r in valid]
    
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, d1, w, label="Top-1 Drop", color="#E74C3C", alpha=0.85)
    ax.bar(x + w/2, d5, w, label="Top-5 Drop", color="#3498DB", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Accuracy Drop", fontsize=10)
    ax.set_title("C. Phase 1: Top-10 Most Important Time-Freq Features", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

def plot_panel_D_phase2(ax, rows):
    """Panel D: Phase 2 扰动总结（对应 fig6）"""
    conds = sorted(set(r["condition"] for r in rows if r.get("condition")))
    if not conds:
        ax.text(0.5, 0.5, "No Phase 2 Data", ha="center", va="center")
        ax.set_title("D. Phase 2: Perturbation Summary")
        return
        
    pert_info = [
        ("scaling",        0.0,   "Scaling (alpha=0.0)"),
        ("phase_rand",     1.0,   "Phase Rand (ratio=1.0)"),
        ("gaussian_noise", -10.0, "Noise (SNR=-10dB)"),
    ]
    x = np.arange(len(conds))
    w = 0.25
    
    for pi, (pt, ep, pl) in enumerate(pert_info):
        drops = []
        for cond in conds:
            m = [r for r in rows if r["condition"] == cond and r["perturbation"] == pt
                 and abs(float(r["param_value"]) - ep) < 0.01]
            drops.append(float(m[0]["top1_drop"]) if m else 0.0)
        ax.bar(x + (pi - 1) * w, drops, w, label=pl, color=COLORS[pi], alpha=0.85)
        
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("__", "\n") for c in conds], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Top-1 Accuracy Drop", fontsize=10)
    ax.set_title("D. Phase 2: Perturbation Types at Max Strength", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "Fig3_Temporal_Composite.png"
    
    print(f"Loading data...")
    p1_rows = load_csv(PHASE1_CSV)
    p2_rows = load_csv(PHASE2_CSV)
    p3_rows = load_csv(PHASE3_CSV)
    sig_map = load_mcnemar_fdr(MCNEMAR_CSV)
    
    p1_data, p1_base = load_phase1_data(p1_rows)
    
    print(f"Plotting composite figure...")
    fig = plt.figure(figsize=(18, 12))
    
    axA = plt.subplot(2, 2, 1)
    plot_panel_A_phase3(axA, p3_rows)
    
    axB = plt.subplot(2, 2, 2)
    plot_panel_B_heatmap(axB, p1_data, p1_base, sig_map)
    
    axC = plt.subplot(2, 2, 3)
    plot_panel_C_top10(axC, p1_rows)
    
    axD = plt.subplot(2, 2, 4)
    plot_panel_D_phase2(axD, p2_rows)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Success! Composite figure saved to {out_file}")

if __name__ == "__main__":
    main()
