"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
========================================
Visualize the dataset to understand:
- Class distribution (sepsis vs non-sepsis)
- Missing value heatmap
- Vital sign distributions
- Correlation heatmap
- Temporal trends of vitals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_DIR    = "./data"
OUTPUT_DIR  = "./eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
LAB_COLS   = ['Glucose', 'Lactate', 'WBC', 'Creatinine', 'Platelets',
              'Potassium', 'Calcium', 'pH']

# ──────────────────────────────────────────────
# Load a sample of patient files for EDA
# ──────────────────────────────────────────────
def load_sample(data_dir, n=500):
    files = [f for f in os.listdir(data_dir) if f.endswith('.psv')][:n]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), sep='|')
        df['PatientID'] = f.replace('.psv', '')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_class_distribution(df):
    """Bar chart — how many sepsis vs non-sepsis patients"""
    patient_labels = df.groupby('PatientID')['SepsisLabel'].max()
    counts = patient_labels.value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Distribution (Patient Level)', fontsize=14, fontweight='bold')

    # Bar chart
    axes[0].bar(['No Sepsis', 'Sepsis'], counts.values,
                color=['steelblue', 'tomato'], edgecolor='black', width=0.5)
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=['No Sepsis', 'Sepsis'],
                colors=['steelblue', 'tomato'], autopct='%1.1f%%',
                startangle=90, wedgeprops=dict(edgecolor='white'))
    axes[1].set_title('Proportion')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_class_distribution.png", dpi=150)
    plt.show()
    print("✅ Plot 1: Class distribution saved")


def plot_missing_values(df):
    """Heatmap of missing value percentage per column"""
    cols_to_check = VITAL_COLS + LAB_COLS
    missing_pct = df[cols_to_check].isnull().mean() * 100
    missing_df  = missing_pct.sort_values(ascending=False).to_frame('Missing %')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d73027' if v > 50 else '#fc8d59' if v > 20 else '#91bfdb'
              for v in missing_df['Missing %']]
    bars = ax.barh(missing_df.index, missing_df['Missing %'],
                   color=colors, edgecolor='white')
    ax.set_xlabel('Missing Value Percentage (%)')
    ax.set_title('Missing Value Analysis by Feature', fontweight='bold')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_missing_values.png", dpi=150)
    plt.show()
    print("✅ Plot 2: Missing values saved")


def plot_vital_distributions(df):
    """Box plots comparing vitals for sepsis vs non-sepsis patients"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle('Vital Signs: Sepsis vs Non-Sepsis Distribution',
                 fontsize=14, fontweight='bold')

    for i, col in enumerate(VITAL_COLS):
        if col in df.columns:
            data_0 = df[df['SepsisLabel'] == 0][col].dropna()
            data_1 = df[df['SepsisLabel'] == 1][col].dropna()
            axes[i].boxplot([data_0, data_1], labels=['No Sepsis', 'Sepsis'],
                            patch_artist=True,
                            boxprops=dict(facecolor='steelblue', alpha=0.6),
                            medianprops=dict(color='red', linewidth=2))
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_vital_distributions.png", dpi=150)
    plt.show()
    print("✅ Plot 3: Vital distributions saved")


def plot_correlation_heatmap(df):
    """Correlation heatmap among key features"""
    cols = VITAL_COLS + ['WBC', 'Lactate', 'Creatinine', 'pH', 'SepsisLabel']
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_correlation_heatmap.png", dpi=150)
    plt.show()
    print("✅ Plot 4: Correlation heatmap saved")


def plot_temporal_trends(df):
    """
    Average vital sign trends over ICU hours
    Compares sepsis vs non-sepsis patients over time
    """
    df_grouped = df.groupby(['ICULOS', 'SepsisLabel'])[VITAL_COLS].mean().reset_index()

    vitals_to_plot = ['HR', 'SBP', 'O2Sat', 'Resp']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle('Average Vital Trends Over ICU Hours (Sepsis vs Non-Sepsis)',
                 fontsize=13, fontweight='bold')

    for i, col in enumerate(vitals_to_plot):
        for label, color, name in [(0, 'steelblue', 'No Sepsis'),
                                   (1, 'tomato', 'Sepsis')]:
            subset = df_grouped[df_grouped['SepsisLabel'] == label]
            axes[i].plot(subset['ICULOS'], subset[col],
                         label=name, color=color, linewidth=2)
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_xlabel('ICU Hours')
        axes[i].set_ylabel('Mean Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 72)   # first 72 hours

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_temporal_trends.png", dpi=150)
    plt.show()
    print("✅ Plot 5: Temporal trends saved")


def plot_lab_comparison(df):
    """Violin plots for key lab values"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle('Lab Values: Sepsis vs Non-Sepsis', fontsize=14, fontweight='bold')

    for i, col in enumerate(LAB_COLS):
        if col in df.columns:
            data = df[['SepsisLabel', col]].dropna()
            data['Group'] = data['SepsisLabel'].map({0: 'No Sepsis', 1: 'Sepsis'})
            sns.violinplot(data=data, x='Group', y=col,
                           palette=['steelblue', 'tomato'], ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_xlabel('')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_lab_comparison.png", dpi=150)
    plt.show()
    print("✅ Plot 6: Lab comparison saved")


def print_summary_stats(df):
    """Print key statistics about the dataset"""
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY STATISTICS")
    print("=" * 50)
    print(f"  Total rows (patient-hours) : {len(df):,}")
    print(f"  Unique patients            : {df['PatientID'].nunique():,}")
    print(f"  Features                   : {df.shape[1]}")
    print(f"  Sepsis cases (rows)        : {df['SepsisLabel'].sum():,}")
    print(f"  Average ICU stay (hours)   : {df.groupby('PatientID')['ICULOS'].max().mean():.1f}")
    print(f"  Average age                : {df['Age'].mean():.1f}")
    print("=" * 50)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print("\nLoading sample data (500 patients)...")
    df = load_sample(DATA_DIR, n=500)
    print(f"Loaded shape: {df.shape}")

    print_summary_stats(df)

    print("\nGenerating plots...")
    plot_class_distribution(df)
    plot_missing_values(df)
    plot_vital_distributions(df)
    plot_correlation_heatmap(df)
    plot_temporal_trends(df)
    plot_lab_comparison(df)

    print(f"\n✅ All EDA plots saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()