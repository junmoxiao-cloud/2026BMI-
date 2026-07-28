import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

def load_json(path: Path):
    if not path or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_mcnemar_and_fdr(phase1_json, phase2_json, phase3_json, output_csv):
    p1_data = load_json(phase1_json)
    p2_data = load_json(phase2_json)
    p3_data = load_json(phase3_json)

    tests_to_run = []

    # 1. Phase 1 (STFT)
    if p1_data:
        results = p1_data.get("results", [])
        baseline = next((r for r in results if r.get("name") == "baseline"), None)
        if baseline and baseline.get("raw_preds"):
            base_preds = baseline["raw_preds"]
            for r in results:
                name = r.get("name")
                if name == "baseline":
                    continue
                cond_preds = r.get("raw_preds")
                if cond_preds and len(cond_preds) == len(base_preds):
                    tests_to_run.append({
                        "Condition": name,
                        "Base_Preds": base_preds,
                        "Cond_Preds": cond_preds,
                        "Top1_Drop": r.get("top1_drop", 0)
                    })

    # 2. Phase 3 (Full Freq Masking)
    if p3_data:
        results = p3_data.get("results", [])
        baseline = p3_data.get("baseline", {})
        if not baseline or not baseline.get("raw_preds"):
            baseline = next((r for r in results if r.get("name") == "baseline"), None)
        
        if baseline and baseline.get("raw_preds"):
            base_preds = baseline["raw_preds"]
            for r in results:
                name = r.get("name")
                if name == "baseline":
                    continue
                cond_preds = r.get("raw_preds")
                if cond_preds and len(cond_preds) == len(base_preds):
                    tests_to_run.append({
                        "Condition": f"Phase3_{name}",
                        "Base_Preds": base_preds,
                        "Cond_Preds": cond_preds,
                        "Top1_Drop": r.get("top1_drop", 0)
                    })

    # 3. Phase 2 (Amplitude Ablation)
    if p2_data:
        results = p2_data.get("results", [])
        overall_baseline = p2_data.get("baseline", {})
        
        # Group by condition and perturbation
        grouped = defaultdict(list)
        for r in results:
            key = (r.get("condition"), r.get("perturbation"))
            grouped[key].append(r)
            
        for (cond, pert), perts in grouped.items():
            base_r = None
            if pert == "scaling":
                base_r = next((x for x in perts if x.get("param_value") == 1.0), None)
            elif pert == "phase_rand":
                base_r = next((x for x in perts if x.get("param_value") == 0.0), None)
            
            if base_r and base_r.get("raw_preds"):
                base_preds = base_r["raw_preds"]
            elif overall_baseline and overall_baseline.get("raw_preds"):
                base_preds = overall_baseline["raw_preds"]
            else:
                continue
                
            for r in perts:
                if base_r and r == base_r:
                    continue
                cond_preds = r.get("raw_preds")
                if cond_preds and len(cond_preds) == len(base_preds):
                    cond_name = f"{cond}_{pert}_{r['param_name']}={r['param_value']}"
                    tests_to_run.append({
                        "Condition": cond_name,
                        "Base_Preds": base_preds,
                        "Cond_Preds": cond_preds,
                        "Top1_Drop": r.get("top1_drop", 0)
                    })

    if not tests_to_run:
        print("No conditions to test.")
        return

    pvals = []
    stats_info = []

    for test in tests_to_run:
        b_preds = test["Base_Preds"]
        c_preds = test["Cond_Preds"]
        
        table = [[0, 0], [0, 0]]
        for b, c in zip(b_preds, c_preds):
            if b and c: table[0][0] += 1
            elif b and not c: table[0][1] += 1
            elif not b and c: table[1][0] += 1
            else: table[1][1] += 1
            
        mc_res = mcnemar(table, exact=True)
        pval = mc_res.pvalue
        pvals.append(pval)
        
        stats_info.append({
            "Condition": test["Condition"],
            "Both_Correct": table[0][0],
            "Base_Correct_Cond_Incorrect": table[0][1],
            "Base_Incorrect_Cond_Correct": table[1][0],
            "Both_Incorrect": table[1][1],
            "Top1_Drop": test["Top1_Drop"],
            "p_value_raw": pval
        })

    # FDR Correction
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    for i, info in enumerate(stats_info):
        info["p_value_fdr"] = pvals_corrected[i]
        info["Significant_FDR_0.05"] = reject[i]

    # Sort by Top1_Drop descending
    stats_info.sort(key=lambda x: x["Top1_Drop"], reverse=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Condition", "Top1_Drop", "Both_Correct", "Base_Correct_Cond_Incorrect", 
        "Base_Incorrect_Cond_Correct", "Both_Incorrect", "p_value_raw", 
        "p_value_fdr", "Significant_FDR_0.05"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_info)

    print(f"Statistical testing completed. Saved report to: {output_csv}")
    print(f"Number of significant conditions (FDR < 0.05): {sum(reject)}")

def main():
    parser = argparse.ArgumentParser(description="Statistical Testing for Temporal Ablation")
    parser.add_argument("--phase1", type=Path, help="Phase 1 JSON")
    parser.add_argument("--phase2", type=Path, help="Phase 2 JSON")
    parser.add_argument("--phase3", type=Path, help="Phase 3 JSON")
    parser.add_argument("--output", type=Path, help="Output CSV report path", required=True)
    parser.add_argument("--input", type=Path, help="Input JSON file (backward compatibility)")
    
    args = parser.parse_args()
    
    if args.input and not args.phase1:
        args.phase1 = args.input

    calculate_mcnemar_and_fdr(args.phase1, args.phase2, args.phase3, args.output)

if __name__ == "__main__":
    main()
