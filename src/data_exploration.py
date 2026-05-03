"""
data_exploration.py
-------------------
Exploratory data analysis of the UCI Iris dataset.

Generates:
  1. Pairplot — all feature combinations coloured by species
  2. Box plots — feature distributions per species
  3. Correlation heatmap — feature relationships
  4. Species statistics summary

Inputs : data/iris.csv
Outputs: results/pairplot.png
         results/feature_boxplots.png
         results/correlation_heatmap.png

Usage:
    python src/data_exploration.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(ROOT, "data", "iris.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

SPECIES_COLOURS = {
    "setosa":     "#378ADD",
    "versicolor": "#1D9E75",
    "virginica":  "#D85A30",
}
FEATURES = ["sepal_length_cm", "sepal_width_cm",
            "petal_length_cm", "petal_width_cm"]
FEATURE_LABELS = ["Sepal Length (cm)", "Sepal Width (cm)",
                  "Petal Length (cm)", "Petal Width (cm)"]


def plot_pairplot(df, output_path):
    """Scatter matrix — all feature pairs coloured by species."""
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle("Iris Dataset — Feature Pairplot\n"
                 "UCI Iris Dataset · Fisher (1936)",
                 fontsize=13, fontweight="bold")

    for i, (fi, li) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        for j, (fj, lj) in enumerate(zip(FEATURES, FEATURE_LABELS)):
            ax = axes[i, j]
            if i == j:
                # Diagonal — KDE per species
                for sp, col in SPECIES_COLOURS.items():
                    subset = df[df.species == sp][fi]
                    ax.hist(subset, bins=12, alpha=0.6,
                            color=col, edgecolor="white", label=sp)
                ax.set_facecolor("#FAFAFA")
            else:
                for sp, col in SPECIES_COLOURS.items():
                    sub = df[df.species == sp]
                    ax.scatter(sub[fj], sub[fi], color=col,
                               alpha=0.6, s=18, edgecolors="none")
                ax.set_facecolor("#FAFAFA")
                ax.grid(True, alpha=0.2)

            if i == 3: ax.set_xlabel(lj, fontsize=8)
            if j == 0: ax.set_ylabel(li, fontsize=8)
            ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s.capitalize())
               for s, c in SPECIES_COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_boxplots(df, output_path):
    """Box plots of each feature per species."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle("Feature Distributions by Species — Box Plots",
                 fontsize=12, fontweight="bold")

    colours = list(SPECIES_COLOURS.values())
    species = list(SPECIES_COLOURS.keys())

    for ax, feat, label in zip(axes, FEATURES, FEATURE_LABELS):
        data = [df[df.species == sp][feat].values for sp in species]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels([s.capitalize() for s in species],
                           fontsize=9, rotation=15)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label.split(" (")[0], fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correlation(df, output_path):
    """Correlation heatmap of numerical features."""
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                ax=ax, annot_kws={"size": 11})
    ax.set_xticklabels(FEATURE_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_LABELS, rotation=0, fontsize=9)
    ax.set_title("Feature Correlation Matrix — Iris Dataset",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("Iris Dataset — Exploratory Analysis")
    print(f"  Samples: {len(df)}  Features: 4  Classes: 3")
    print(f"\n  Species distribution:")
    for sp, count in df.species.value_counts().items():
        print(f"    {sp:<12} : {count} samples")

    print(f"\n  Feature statistics:")
    print(df[FEATURES].describe().round(3).to_string())

    print("\nGenerating plots...")
    plot_pairplot(df, os.path.join(RESULTS_DIR, "pairplot.png"))
    plot_boxplots(df, os.path.join(RESULTS_DIR, "feature_boxplots.png"))
    plot_correlation(df, os.path.join(RESULTS_DIR, "correlation_heatmap.png"))

    # Key separability insight
    print("\n  Separability insight:")
    for feat in FEATURES:
        ranges = {sp: (df[df.species==sp][feat].min(),
                       df[df.species==sp][feat].max())
                  for sp in SPECIES_COLOURS}
        print(f"    {feat:<22}: " +
              "  ".join([f"{sp}=[{r[0]:.1f},{r[1]:.1f}]"
                         for sp, r in ranges.items()]))

    print("\nData exploration complete.")


if __name__ == "__main__":
    main()
