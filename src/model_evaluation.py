"""
model_evaluation.py
-------------------
Evaluate all five classifiers on the held-out test set.

Generates:
  1. Confusion matrices — all five models (2x3 panel)
  2. Classification report — precision, recall, F1 per class
  3. ROC curves — one-vs-rest for each classifier

Inputs : data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
Outputs: results/confusion_matrices.png
         results/classification_report.png
         results/roc_curves.png

Usage:
    python src/model_evaluation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#BA7517"]


def load_data():
    return (np.load(os.path.join(DATA_DIR,"X_train.npy")),
            np.load(os.path.join(DATA_DIR,"X_test.npy")),
            np.load(os.path.join(DATA_DIR,"y_train.npy")),
            np.load(os.path.join(DATA_DIR,"y_test.npy")),
            np.load(os.path.join(DATA_DIR,"class_names.npy"),
                    allow_pickle=True))


def rebuild_models(X_tr, y_tr):
    with open(os.path.join(MODELS_DIR,"training_summary.json")) as f:
        s = json.load(f)

    def p(name, key, default):
        v = s[name]["params"].get(key, str(default))
        if v == "None": return None
        try:
            return int(v)
        except (ValueError, TypeError):
            try:
                return float(v)
            except (ValueError, TypeError):
                return v

    models = {
        "KNN": KNeighborsClassifier(
            n_neighbors=p("KNN","n_neighbors",5),
            weights=s["KNN"]["params"].get("weights","uniform"),
            metric=s["KNN"]["params"].get("metric","euclidean")),
        "SVM": SVC(kernel="rbf", probability=True,
            C=p("SVM","C",1.0),
            gamma=s["SVM"]["params"].get("gamma","scale"),
            random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=p("Decision Tree","max_depth",None),
            min_samples_split=p("Decision Tree","min_samples_split",2),
            criterion=s["Decision Tree"]["params"].get("criterion","gini"),
            random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=p("Random Forest","n_estimators",100),
            max_depth=p("Random Forest","max_depth",None),
            min_samples_split=p("Random Forest","min_samples_split",2),
            random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(
            C=p("Logistic Regression","C",1.0),
            solver=s["Logistic Regression"]["params"].get("solver","lbfgs"),
            max_iter=1000, random_state=RANDOM_SEED),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models


def plot_confusion_matrices(models, X_te, y_te, classes, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Confusion Matrices — Test Set (30 samples)",
                 fontsize=13, fontweight="bold")

    for ax, (name, model), colour in zip(
            list(axes.flat)[:5], models.items(), COLOURS):
        cm = confusion_matrix(y_te, model.predict(X_te))
        acc = cm.diagonal().sum() / cm.sum()
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i,j]),
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()*0.6 else "black")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([c.capitalize() for c in classes],
                           rotation=15, fontsize=9)
        ax.set_yticklabels([c.capitalize() for c in classes], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"{name}\nAccuracy = {acc:.4f}",
                     fontsize=11, fontweight="bold")

    axes.flat[5].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_classification_report(models, X_te, y_te, classes, output_path):
    """Heatmap of precision, recall, F1 per model per class."""
    metrics_list = []
    for name, model in models.items():
        y_pred = model.predict(X_te)
        for i, cls in enumerate(classes):
            tp = ((y_pred==i)&(y_te==i)).sum()
            fp = ((y_pred==i)&(y_te!=i)).sum()
            fn = ((y_pred!=i)&(y_te==i)).sum()
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            metrics_list.append({"Model":name,"Class":cls.capitalize(),
                                  "Precision":prec,"Recall":rec,"F1":f1})

    import pandas as pd
    df = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Classification Report — Precision / Recall / F1 per Class",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, ["Precision","Recall","F1"]):
        pivot = df.pivot(index="Model", columns="Class", values=metric)
        im = ax.imshow(pivot.values, cmap="Greens", vmin=0.8, vmax=1.0)
        plt.colorbar(im, ax=ax, shrink=0.85)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if pivot.values[i,j]>0.95 else "black")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(metric, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_curves(models, X_te, y_te, classes, output_path):
    """One-vs-rest ROC curves for each model."""
    y_bin = label_binarize(y_te, classes=range(len(classes)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("ROC Curves — One-vs-Rest per Classifier",
                 fontsize=13, fontweight="bold")
    cls_colours = ["#378ADD","#1D9E75","#D85A30"]

    for ax, (name, model), m_col in zip(
            list(axes.flat)[:5], models.items(), COLOURS):
        y_prob = model.predict_proba(X_te)
        for i, (cls, col) in enumerate(zip(classes, cls_colours)):
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, linewidth=2.0,
                    label=f"{cls.capitalize()} (AUC={roc_auc:.3f})")
        ax.plot([0,1],[0,1],"k--",linewidth=0.8,alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, alpha=0.2)

    axes.flat[5].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Model Evaluation — Iris Species Classification\n")

    X_tr, X_te, y_tr, y_te, classes = load_data()
    models = rebuild_models(X_tr, y_tr)

    print(f"  {'Model':<22} {'Accuracy':>9}")
    print("  " + "─" * 34)
    for name, model in models.items():
        acc = (model.predict(X_te) == y_te).mean()
        print(f"  {name:<22} {acc:>9.4f}")

    print("\nGenerating plots...")
    plot_confusion_matrices(models, X_te, y_te, classes,
        os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_classification_report(models, X_te, y_te, classes,
        os.path.join(RESULTS_DIR, "classification_report.png"))
    plot_roc_curves(models, X_te, y_te, classes,
        os.path.join(RESULTS_DIR, "roc_curves.png"))

    print("\nModel evaluation complete.")


if __name__ == "__main__":
    main()
