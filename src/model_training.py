"""
model_training.py
-----------------
Train and tune five classifiers on the Iris dataset.

Models:
  1. K-Nearest Neighbours (KNN)
  2. Support Vector Machine (SVM) with RBF kernel
  3. Decision Tree
  4. Random Forest
  5. Logistic Regression

Each model tuned via 5-fold stratified cross-validation GridSearchCV.
Results saved to data/models/training_summary.json.

Inputs : data/iris.csv
Outputs: data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
         data/models/training_summary.json
         results/cv_accuracy_comparison.png

Usage:
    python src/model_training.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(ROOT, "data", "iris.csv")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]


def load_and_split(path):
    df  = pd.read_csv(path)
    le  = LabelEncoder()
    X   = df[["sepal_length_cm","sepal_width_cm",
              "petal_length_cm","petal_width_cm"]].values
    y   = le.fit_transform(df["species"].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_tr)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_te)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_te)
    np.save(os.path.join(DATA_DIR, "class_names.npy"), le.classes_)

    return X_tr, X_te, y_tr, y_te, le.classes_


def train_models(X_tr, y_tr):
    configs = [
        ("KNN",
         KNeighborsClassifier(),
         {"n_neighbors": [3,5,7,9,11],
          "weights": ["uniform","distance"],
          "metric":  ["euclidean","manhattan"]}),
        ("SVM",
         SVC(kernel="rbf", random_state=RANDOM_SEED, probability=True),
         {"C": [0.1,1,10,100], "gamma": ["scale","auto"]}),
        ("Decision Tree",
         DecisionTreeClassifier(random_state=RANDOM_SEED),
         {"max_depth": [None,3,4,5],
          "min_samples_split": [2,5],
          "criterion": ["gini","entropy"]}),
        ("Random Forest",
         RandomForestClassifier(random_state=RANDOM_SEED),
         {"n_estimators": [50,100,200],
          "max_depth": [None,3,5],
          "min_samples_split": [2,5]}),
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
         {"C": [0.01,0.1,1,10,100],
          "solver": ["lbfgs","newton-cg","saga"]}),
    ]

    results = {}
    for name, estimator, params in configs:
        print(f"  [{len(results)+1}/5] {name}...")
        gs = GridSearchCV(estimator, params, cv=5,
                          scoring="accuracy", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        cv_scores = cross_val_score(gs.best_estimator_, X_tr, y_tr,
                                    cv=5, scoring="accuracy")
        results[name] = {
            "model":       gs.best_estimator_,
            "cv_mean":     cv_scores.mean(),
            "cv_std":      cv_scores.std(),
            "best_params": gs.best_params_,
        }
        print(f"     CV Acc = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
              f"params={gs.best_params_}")

    return results


def plot_cv_comparison(results, output_path):
    names  = list(results.keys())
    means  = [results[n]["cv_mean"] for n in names]
    stds   = [results[n]["cv_std"]  for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, means, color=COLOURS, width=0.5, edgecolor="white")
    ax.errorbar(names, means, yerr=stds, fmt="none",
                color="black", capsize=6, linewidth=1.5)
    ax.bar_label(bars, fmt="%.4f", padding=5,
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Cross-Validation Accuracy (5-fold)", fontsize=11)
    ax.set_title("Classifier Comparison — 5-Fold CV Accuracy\n"
                 "Iris Species Classification · Bharat Intern ML Internship 2023",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.85, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":",
               alpha=0.6, label="Perfect accuracy")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Model Training — Iris Species Classification\n")
    X_tr, X_te, y_tr, y_te, classes = load_and_split(DATA_PATH)
    print(f"  Train: {X_tr.shape[0]} samples  Test: {X_te.shape[0]} samples")
    print(f"  Classes: {list(classes)}\n")

    results = train_models(X_tr, y_tr)

    # Save summary
    summary = {n: {
        "cv_mean": round(r["cv_mean"], 4),
        "cv_std":  round(r["cv_std"],  4),
        "params":  {k: str(v) for k,v in r["best_params"].items()}
    } for n, r in results.items()}

    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_cv_comparison(results,
        os.path.join(RESULTS_DIR, "cv_accuracy_comparison.png"))

    best = max(results, key=lambda k: results[k]["cv_mean"])
    print(f"\n  Best model : {best}")
    print(f"  Best CV Acc: {results[best]['cv_mean']:.4f}")
    print("\nModel training complete.")


if __name__ == "__main__":
    main()
