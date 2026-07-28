import argparse
import json
import csv
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

def calculate_mcnemar_and_fdr(input_json: Path, output_csv: Path):
    if not input_json.exists():
        print(f"Error: {input_json} does not exist.")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    
    # Find baseline
    baseline = None
    for r in results:
        if r.get("name") == "baseline":
            baseline = r
            break

    if not baseline:
        print("Error: Could not find 'baseline' condition in the results.")
        return

    base_preds = baseline.get("raw_preds")
    if not base_preds:
        print("Error: Baseline does not have 'raw_preds'.")
        return

    conditions = []
    pvals = []
    stats_info = []

    for r in results:
        name = r.get("name")
        if name == "baseline":
            continue

        cond_preds = r.get("raw_preds")
        if not cond_preds:
            print(f"Warning: Condition '{name}' does not have 'raw_preds'. Skipping.")
            continue

        if len(base_preds) != len(cond_preds):
            print(f"Warning: Length mismatch for '{name}'. Skipping.")
            continue

        # Build 2x2 contingency table
        # table[0][0]: both correct
        # table[0][1]: baseline correct, condition incorrect (discordant)
        # table[1][0]: baseline incorrect, condition correct (discordant)
        # table[1][1]: both incorrect
        table = [[0, 0], [0, 0]]
        for b, c in zip(base_preds, cond_preds):
            if b and c:
                table[0][0] += 1
            elif b and not c:
                table[0][1] += 1
            elif not b and c:
                table[1][0] += 1
            else:
                table[1][1] += 1

        # McNemar test
        # exact=True computes binomial exact p-value (good for small/large N)
        mc_res = mcnemar(table, exact=True)
        pval = mc_res.pvalue

        conditions.append(name)
        pvals.append(pval)
        
        # Calculate accuracy drop for reference
        top1_drop = r.get("top1_drop")
        if top1_drop is None:
            top1_drop = round((sum(base_preds) - sum(cond_preds)) / len(base_preds), 4)

        stats_info.append({
            "Condition": name,
            "Both_Correct": table[0][0],
            "Base_Correct_Cond_Incorrect": table[0][1],
            "Base_Incorrect_Cond_Correct": table[1][0],
            "Both_Incorrect": table[1][1],
            "Top1_Drop": top1_drop,
            "p_value_raw": pval
        })

    if not pvals:
        print("No conditions to test.")
        return

    # FDR Correction
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    for i, info in enumerate(stats_info):
        info["p_value_fdr"] = pvals_corrected[i]
        info["Significant_FDR_0.05"] = reject[i]

    # Sort by Top1_Drop descending
    stats_info.sort(key=lambda x: x["Top1_Drop"], reverse=True)

    # Save to CSV
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
    parser.add_argument("--input", type=Path, help="Input JSON file containing raw_preds", required=True)
    parser.add_argument("--output", type=Path, help="Output CSV report path", required=True)
    args = parser.parse_args()

    calculate_mcnemar_and_fdr(args.input, args.output)

if __name__ == "__main__":
    main()
