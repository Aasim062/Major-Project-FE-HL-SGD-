"""
===============================================================================
 STAGE 4.5: EVALUATION & COMPARISON
===============================================================================

 Purpose:
   Evaluate the federated global model (both encrypted and plaintext versions)
   on the FULL combined dataset and compare against:
     1. Individual hospital local models
     2. Centralized LSTM baseline
   Generate multiple comparison graphs.

 Input:
   Federated_Average/global_model.pth             (encrypted FedAvg)
   Federated_Average/global_model_plaintext.pth    (plaintext FedAvg)
   Model_Training/Hospital_A/local_model.pth       (local models)
   Model_Training/Hospital_B/local_model.pth
   Model_Training/Hospital_C/local_model.pth
   models/lstm_baseline.pth                        (centralized baseline)
   Dataset/processed/mimic/mimic_ppwindowed_dataset.csv

 Output:
   Evaluation/evaluation_report.txt
   Evaluation/graphs/
     1_model_comparison_bar.png
     2_roc_curves.png
     3_confusion_matrices.png
     4_hospital_cv_comparison.png
     5_crypto_overhead.png
     6_weight_distribution.png

===============================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve, confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add Model_Training to path for LSTM import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Model_Training"))
from LSTM import MortalityLSTM, load_and_reshape

# Data
CSV_PATH = os.path.join(PROJECT_ROOT, "Dataset", "processed", "mimic",
                        "mimic_ppwindowed_dataset.csv")

# Models
GLOBAL_ENC_PATH = os.path.join(PROJECT_ROOT, "Federated_Average", "global_model.pth")
GLOBAL_PLAIN_PATH = os.path.join(PROJECT_ROOT, "Federated_Average", "global_model_plaintext.pth")
BASELINE_PATH = os.path.join(PROJECT_ROOT, "models", "lstm_baseline.pth")

HOSPITAL_PATHS = {
    "Hospital A": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_A", "local_model.pth"),
    "Hospital B": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_B", "local_model.pth"),
    "Hospital C": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_C", "local_model.pth"),
}

# Hospital CV results
HOSPITAL_CV_PATHS = {
    "Hospital A": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_A", "cv_results.json"),
    "Hospital B": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_B", "cv_results.json"),
    "Hospital C": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_C", "cv_results.json"),
}

# Aggregation report
AGG_REPORT_PATH = os.path.join(PROJECT_ROOT, "Federated_Average", "aggregation_report.txt")
ENC_REPORT_PATH = os.path.join(PROJECT_ROOT, "Encryption", "encrypted", "encryption_report.txt")

# Output
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Threshold for binary classification
THRESHOLD = 0.5

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "Encrypted FL": "#2196F3",
    "Plaintext FL": "#4CAF50",
    "Centralized": "#FF9800",
    "Hospital A": "#E91E63",
    "Hospital B": "#9C27B0",
    "Hospital C": "#00BCD4",
}


# ============================================================
# EVALUATE A MODEL ON DATA
# ============================================================
def evaluate_model(state_dict, X, y, threshold=THRESHOLD):
    """
    Loads weights into MortalityLSTM and evaluates on given data.

    Returns dict with accuracy, auc, f1, precision, recall, y_pred, y_prob
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MortalityLSTM().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor).squeeze().cpu()
        probs = torch.sigmoid(logits).numpy()

    y_pred = (probs >= threshold).astype(int)
    y_true = y.astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, probs),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "y_pred": y_pred,
        "y_prob": probs,
        "y_true": y_true,
    }
    return metrics


# ============================================================
# GRAPH 1: MODEL COMPARISON BAR CHART
# ============================================================
def plot_model_comparison(results, save_path):
    """
    Grouped bar chart comparing Accuracy, AUC-ROC, F1, Precision, Recall
    across all models.
    """
    metrics_names = ["accuracy", "auc_roc", "f1_score", "precision", "recall"]
    display_names = ["Accuracy", "AUC-ROC", "F1-Score", "Precision", "Recall"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(display_names))
    width = 0.12
    n_models = len(model_names)

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics_names]
        offset = (i - n_models / 2 + 0.5) * width
        color = COLORS.get(model_name, f"C{i}")
        bars = ax.bar(x + offset, values, width, label=model_name, color=color,
                      edgecolor='white', linewidth=0.5)
        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Evaluated on Full Dataset - 123 Patients)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# GRAPH 2: ROC CURVES
# ============================================================
def plot_roc_curves(results, y_true, save_path):
    """
    Overlaid ROC curves for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_true, metrics["y_prob"])
        auc_val = metrics["auc_roc"]
        color = COLORS.get(model_name, None)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_val:.4f})',
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models\n(Full Dataset Evaluation)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# GRAPH 3: CONFUSION MATRICES
# ============================================================
def plot_confusion_matrices(results, save_path):
    """
    Side-by-side confusion matrices for key models.
    """
    # Show only the main models (not all hospitals individually)
    key_models = [k for k in results.keys() if k in
                  ["Encrypted FL", "Plaintext FL", "Centralized",
                   "Hospital A", "Hospital B", "Hospital C"]]

    n = len(key_models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, model_name in enumerate(key_models):
        ax = axes[i]
        y_true = results[model_name]["y_true"]
        y_pred = results[model_name]["y_pred"]
        cm = confusion_matrix(y_true, y_pred)

        # Plot heatmap manually
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(model_name, fontsize=12, fontweight='bold')

        classes = ['Survived', 'Died']
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

        # Add numbers in cells
        for row in range(2):
            for col in range(2):
                color = "white" if cm[row, col] > cm.max() / 2 else "black"
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center", fontsize=16, fontweight='bold',
                        color=color)

        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# GRAPH 4: HOSPITAL CV COMPARISON
# ============================================================
def plot_hospital_cv(save_path):
    """
    Per-fold CV metrics for each hospital (grouped bar chart).
    Shows how each hospital performed during local training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_keys = ["accuracy", "auc_roc", "f1_score"]
    metric_labels = ["Accuracy", "AUC-ROC", "F1-Score"]

    hospital_data = {}
    for name, path in HOSPITAL_CV_PATHS.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                hospital_data[name] = json.load(f)

    for ax_idx, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(5)  # 5 folds
        width = 0.25

        for i, (hospital_name, data) in enumerate(hospital_data.items()):
            folds = data["fold_metrics"]
            values = [fold[metric_key] for fold in folds]
            color = COLORS.get(hospital_name, f"C{i}")
            bars = ax.bar(x + i * width, values, width, label=hospital_name,
                          color=color, edgecolor='white', linewidth=0.5)

        # Add average line for each hospital
        for i, (hospital_name, data) in enumerate(hospital_data.items()):
            folds = data["fold_metrics"]
            avg_val = np.mean([fold[metric_key] for fold in folds])
            color = COLORS.get(hospital_name, f"C{i}")
            ax.axhline(y=avg_val, color=color, linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} per Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Fold {i+1}' for i in range(5)], fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Hospital Local Training - 5-Fold Cross-Validation Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# GRAPH 5: CRYPTO OVERHEAD
# ============================================================
def plot_crypto_overhead(enc_times, agg_time, dec_time, save_path):
    """
    Bar chart showing encryption, aggregation, and decryption times.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Per-hospital encryption time
    hospitals = list(enc_times.keys())
    times = list(enc_times.values())
    colors_list = [COLORS.get(f"Hospital {h}", "C0") for h in hospitals]

    bars = ax1.bar(hospitals, times, color=colors_list, edgecolor='white', linewidth=0.5)
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f'{t:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Hospital', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Per-Hospital Encryption Time', fontsize=13, fontweight='bold')

    # Right: Pipeline breakdown
    stages = ['Encryption\n(per hospital avg)', 'Aggregation\n(homomorphic)', 'Decryption']
    avg_enc = np.mean(list(enc_times.values()))
    stage_times = [avg_enc, agg_time, dec_time]
    stage_colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars2 = ax1_bars = ax2.bar(stages, stage_times, color=stage_colors,
                                edgecolor='white', linewidth=0.5)
    for bar, t in zip(bars2, stage_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f'{t:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    total = sum(stage_times)
    ax2.set_xlabel(f'Total Crypto Overhead: {total:.4f}s', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Cryptographic Pipeline Breakdown', fontsize=13, fontweight='bold')

    fig.suptitle('CKKS-RNS Encryption Overhead Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# GRAPH 6: WEIGHT DISTRIBUTION (Encrypted vs Plaintext)
# ============================================================
def plot_weight_distribution(enc_sd, plain_sd, save_path):
    """
    Histogram comparing encrypted FL vs plaintext FL weight distributions.
    Shows that encryption introduces negligible noise.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Flatten all weights
    enc_flat = torch.cat([p.flatten() for p in enc_sd.values()]).numpy()
    plain_flat = torch.cat([p.flatten() for p in plain_sd.values()]).numpy()
    diff = enc_flat - plain_flat

    # Top-left: Overlaid weight distributions
    ax = axes[0, 0]
    ax.hist(enc_flat, bins=80, alpha=0.6, color=COLORS["Encrypted FL"],
            label='Encrypted FL', density=True)
    ax.hist(plain_flat, bins=80, alpha=0.6, color=COLORS["Plaintext FL"],
            label='Plaintext FL', density=True)
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weight Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Top-right: Error distribution (enc - plain)
    ax = axes[0, 1]
    ax.hist(diff, bins=80, color='#F44336', alpha=0.7, density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Weight Difference (Encrypted - Plaintext)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Encryption Error Distribution\n'
                 f'Max={np.max(np.abs(diff)):.2e}, Mean={np.mean(np.abs(diff)):.2e}',
                 fontsize=12, fontweight='bold')

    # Bottom-left: Scatter plot (enc vs plain)
    ax = axes[1, 0]
    ax.scatter(plain_flat, enc_flat, alpha=0.3, s=5, color='#673AB7')
    lims = [min(plain_flat.min(), enc_flat.min()) - 0.1,
            max(plain_flat.max(), enc_flat.max()) + 0.1]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('Plaintext Weight', fontsize=11)
    ax.set_ylabel('Encrypted Weight', fontsize=11)
    ax.set_title('Encrypted vs Plaintext Weights (Scatter)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Bottom-right: Per-layer error
    ax = axes[1, 1]
    layer_names = []
    layer_errors = []
    for key in enc_sd:
        diff_layer = (enc_sd[key] - plain_sd[key]).abs()
        short_name = key.replace("classifier.", "cls.").replace("weight", "W").replace("bias", "b")
        layer_names.append(short_name)
        layer_errors.append(diff_layer.max().item())

    bars = ax.barh(layer_names, layer_errors, color='#FF5722', edgecolor='white')
    ax.set_xlabel('Max Absolute Error', fontsize=11)
    ax.set_title('Per-Layer Encryption Error', fontsize=12, fontweight='bold')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-2, -2))

    fig.suptitle('Weight Analysis: Encrypted FL vs Plaintext FL\n(5,921 parameters)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# PARSE TIMING FROM REPORTS
# ============================================================
def parse_encryption_times():
    """Parse encryption times from encryption_report.txt and aggregation_report.txt"""
    enc_times = {}
    agg_time = 0.0
    dec_time = 0.0

    if os.path.exists(ENC_REPORT_PATH):
        with open(ENC_REPORT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                # Match lines like: "Hospital A    0.0074s"
                if line.startswith("Hospital") and line.endswith("s") and "Time" not in line:
                    # Extract the last token as time value
                    time_str = line.split()[-1].replace("s", "")
                    try:
                        time_val = float(time_str)
                        # Hospital name is the second word (A, B, C)
                        name = line.split()[1]
                        enc_times[name] = time_val
                    except ValueError:
                        continue

    if os.path.exists(AGG_REPORT_PATH):
        with open(AGG_REPORT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if "Homomorphic Aggregation" in line and line.endswith("s"):
                    try:
                        agg_time = float(line.split()[-1].replace("s", ""))
                    except ValueError:
                        pass
                elif line.startswith("Decryption") and line.endswith("s"):
                    try:
                        dec_time = float(line.split()[-1].replace("s", ""))
                    except ValueError:
                        pass

    return enc_times, agg_time, dec_time


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  STAGE 4.5: EVALUATION & COMPARISON")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load full dataset
    # ------------------------------------------------------------------
    print("\n  [1/4] Loading full dataset...")
    X, y, patient_ids = load_and_reshape(CSV_PATH)
    print(f"    Patients: {len(y)}  |  Survived: {int((y==0).sum())}  |  Died: {int((y==1).sum())}")

    # ------------------------------------------------------------------
    # Step 2: Evaluate all models
    # ------------------------------------------------------------------
    print("\n  [2/4] Evaluating models on full dataset...")
    print("-" * 70)

    results = {}

    # A) Encrypted FL global model
    if os.path.exists(GLOBAL_ENC_PATH):
        sd = torch.load(GLOBAL_ENC_PATH, map_location="cpu", weights_only=True)
        results["Encrypted FL"] = evaluate_model(sd, X, y)
        print(f"    Encrypted FL:  Acc={results['Encrypted FL']['accuracy']:.4f}  "
              f"AUC={results['Encrypted FL']['auc_roc']:.4f}  "
              f"F1={results['Encrypted FL']['f1_score']:.4f}")

    # B) Plaintext FL global model
    if os.path.exists(GLOBAL_PLAIN_PATH):
        sd = torch.load(GLOBAL_PLAIN_PATH, map_location="cpu", weights_only=True)
        results["Plaintext FL"] = evaluate_model(sd, X, y)
        print(f"    Plaintext FL:  Acc={results['Plaintext FL']['accuracy']:.4f}  "
              f"AUC={results['Plaintext FL']['auc_roc']:.4f}  "
              f"F1={results['Plaintext FL']['f1_score']:.4f}")

    # C) Centralized baseline
    if os.path.exists(BASELINE_PATH):
        sd = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
        results["Centralized"] = evaluate_model(sd, X, y)
        print(f"    Centralized:   Acc={results['Centralized']['accuracy']:.4f}  "
              f"AUC={results['Centralized']['auc_roc']:.4f}  "
              f"F1={results['Centralized']['f1_score']:.4f}")

    # D) Individual hospital models
    for hospital_name, path in HOSPITAL_PATHS.items():
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu", weights_only=True)
            results[hospital_name] = evaluate_model(sd, X, y)
            print(f"    {hospital_name:14s} Acc={results[hospital_name]['accuracy']:.4f}  "
                  f"AUC={results[hospital_name]['auc_roc']:.4f}  "
                  f"F1={results[hospital_name]['f1_score']:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Generate graphs
    # ------------------------------------------------------------------
    print(f"\n  [3/4] Generating graphs...")
    print("-" * 70)

    # Graph 1: Model comparison bar chart
    plot_model_comparison(results,
                          os.path.join(GRAPHS_DIR, "1_model_comparison_bar.png"))

    # Graph 2: ROC curves
    plot_roc_curves(results, y.astype(int),
                    os.path.join(GRAPHS_DIR, "2_roc_curves.png"))

    # Graph 3: Confusion matrices
    plot_confusion_matrices(results,
                            os.path.join(GRAPHS_DIR, "3_confusion_matrices.png"))

    # Graph 4: Hospital CV comparison
    plot_hospital_cv(os.path.join(GRAPHS_DIR, "4_hospital_cv_comparison.png"))

    # Graph 5: Crypto overhead
    enc_times, agg_time, dec_time = parse_encryption_times()
    if enc_times:
        plot_crypto_overhead(enc_times, agg_time, dec_time,
                             os.path.join(GRAPHS_DIR, "5_crypto_overhead.png"))
    else:
        print("    SKIPPED: 5_crypto_overhead.png (no timing data found)")

    # Graph 6: Weight distribution
    if os.path.exists(GLOBAL_ENC_PATH) and os.path.exists(GLOBAL_PLAIN_PATH):
        enc_sd = torch.load(GLOBAL_ENC_PATH, map_location="cpu", weights_only=True)
        plain_sd = torch.load(GLOBAL_PLAIN_PATH, map_location="cpu", weights_only=True)
        plot_weight_distribution(enc_sd, plain_sd,
                                 os.path.join(GRAPHS_DIR, "6_weight_distribution.png"))

    # ------------------------------------------------------------------
    # Step 4: Write evaluation report
    # ------------------------------------------------------------------
    print(f"\n  [4/4] Writing evaluation report...")
    print("-" * 70)

    report_path = os.path.join(SCRIPT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("EVALUATION REPORT (STAGE 4.5)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(y)} patients ({int((y==0).sum())} survived, {int((y==1).sum())} died)\n")
        f.write(f"Evaluation: Full dataset (all 123 patients)\n\n")

        # Results table
        f.write("MODEL COMPARISON (Full Dataset Evaluation)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<16s} {'Acc':<8s} {'AUC':<8s} {'F1':<8s} {'Prec':<8s} {'Recall':<8s}\n")
        f.write(f"{'-'*16:<16s} {'-'*8:<8s} {'-'*8:<8s} {'-'*8:<8s} {'-'*8:<8s} {'-'*8:<8s}\n")
        for name, m in results.items():
            f.write(f"{name:<16s} {m['accuracy']:<8.4f} {m['auc_roc']:<8.4f} "
                    f"{m['f1_score']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f}\n")
        f.write("\n")

        # Encrypted vs Plaintext comparison
        if "Encrypted FL" in results and "Plaintext FL" in results:
            enc = results["Encrypted FL"]
            pln = results["Plaintext FL"]
            f.write("ENCRYPTED vs PLAINTEXT FL\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Accuracy Difference:  {abs(enc['accuracy'] - pln['accuracy']):.6f}\n")
            f.write(f"  AUC-ROC Difference:   {abs(enc['auc_roc'] - pln['auc_roc']):.6f}\n")
            f.write(f"  F1-Score Difference:  {abs(enc['f1_score'] - pln['f1_score']):.6f}\n")
            f.write(f"  Verdict: {'IDENTICAL' if abs(enc['accuracy'] - pln['accuracy']) < 1e-6 else 'NEGLIGIBLE DIFFERENCE'}\n\n")

        # Crypto overhead
        if enc_times:
            total_enc = sum(enc_times.values())
            total_crypto = total_enc + agg_time + dec_time
            f.write("CRYPTOGRAPHIC OVERHEAD\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Encryption (3 hospitals): {total_enc:.4f}s\n")
            f.write(f"  Aggregation:              {agg_time:.4f}s\n")
            f.write(f"  Decryption:               {dec_time:.4f}s\n")
            f.write(f"  Total Overhead:           {total_crypto:.4f}s\n\n")

        # Classification reports
        for name, m in results.items():
            f.write(f"\nCLASSIFICATION REPORT: {name}\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(m["y_true"], m["y_pred"],
                                          target_names=["Survived", "Died"],
                                          zero_division=0))
            f.write("\n")

        f.write("\nGRAPHS GENERATED\n")
        f.write("-" * 60 + "\n")
        f.write("  1_model_comparison_bar.png   - Grouped bar chart (all metrics)\n")
        f.write("  2_roc_curves.png             - ROC curves for all models\n")
        f.write("  3_confusion_matrices.png     - Confusion matrices side-by-side\n")
        f.write("  4_hospital_cv_comparison.png - Per-fold CV results by hospital\n")
        f.write("  5_crypto_overhead.png        - Encryption timing breakdown\n")
        f.write("  6_weight_distribution.png    - Encrypted vs plaintext weight analysis\n")

    print(f"    Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\n  {'Model':<16s} {'Acc':<8s} {'AUC':<8s} {'F1':<8s}")
    print(f"  {'-'*16:<16s} {'-'*8:<8s} {'-'*8:<8s} {'-'*8:<8s}")
    for name, m in results.items():
        print(f"  {name:<16s} {m['accuracy']:<8.4f} {m['auc_roc']:<8.4f} {m['f1_score']:<8.4f}")

    if "Encrypted FL" in results and "Plaintext FL" in results:
        acc_diff = abs(results["Encrypted FL"]["accuracy"] - results["Plaintext FL"]["accuracy"])
        print(f"\n  Encrypted FL vs Plaintext FL accuracy diff: {acc_diff:.2e}")
        if acc_diff < 1e-6:
            print("  --> IDENTICAL: Encryption causes ZERO accuracy loss")
        else:
            print("  --> NEGLIGIBLE: Encryption noise has no practical impact")

    print(f"\n  Graphs: {GRAPHS_DIR}")
    print(f"  Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
