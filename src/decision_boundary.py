"""
decision_boundary.py
--------------------
Decision boundary visualisation for all five classifiers.

Projects the 4D feature space to 2D using the two most discriminative
feature pairs (petal length vs petal width, sepal length vs petal length),
then plots colour-filled decision regions with training and test points.

This is the most visually impactful plot in the repo — it shows at a
glance how each classifier partitions the feature space.

Inputs : data/iris.csv (for raw plotting)
         data/X_train.npy, y_train.npy (for fitting)
Outputs: results/decision_boundaries.png

Usage:
    python src/decision_boundary.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42

SPECIES_COLOURS = ["#378ADD","#1D9E75","#D85A30"]
BG_COLOURS      = ["#D6E8F8","#C8EFE2","#F8DDD4"]


def build_models():
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

    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=p("KNN","n_neighbors",5),
            weights=s["KNN"]["params"].get("weights","uniform")),
        "SVM": SVC(kernel="rbf",
            C=p("SVM","C",1.0),
            gamma=s["SVM"]["params"].get("gamma","scale"),
            random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=p("Decision Tree","max_depth",None),
            random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=p("Random Forest","n_estimators",100),
            random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(
            C=p("Logistic Regression","C",1.0),
            max_iter=1000, random_state=RANDOM_SEED),
    }


def plot_decision_boundaries(output_path):
    df = pd.read_csv(os.path.join(ROOT, "data", "iris.csv"))
    species_map = {"setosa":0,"versicolor":1,"virginica":2}
    y = df["species"].map(species_map).values

    # Use petal length (2) and petal width (3) — most discriminative pair
    feat_idx = [2, 3]
    feat_labels = ["Petal Length (cm)", "Petal Width (cm)"]
    X2 = df[["petal_length_cm","petal_width_cm"]].values

    scaler = StandardScaler()
    X2_sc  = scaler.fit_transform(X2)

    models = build_models()
    for m in models.values():
        m.fit(X2_sc, y)

    # Mesh
    x_min, x_max = X2_sc[:,0].min()-0.5, X2_sc[:,0].max()+0.5
    y_min, y_max = X2_sc[:,1].min()-0.5, X2_sc[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300),
                         np.linspace(y_min,y_max,300))

    cmap_bg = ListedColormap(BG_COLOURS)
    markers = ["o","s","^"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Decision Boundaries — Petal Length vs Petal Width\n"
                 "Five Classifiers · Iris Species Classification",
                 fontsize=13, fontweight="bold")

    for ax, (name, model) in zip(list(axes.flat)[:5], models.items()):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.75)
        ax.contour(xx, yy, Z, colors="white", linewidths=0.6, alpha=0.5)

        for cls_id, (sp, col, mk) in enumerate(
                zip(["setosa","versicolor","virginica"],
                    SPECIES_COLOURS, markers)):
            mask = y == cls_id
            Xm   = X2_sc[mask]
            ax.scatter(Xm[:,0], Xm[:,1], color=col, marker=mk,
                       s=40, edgecolors="black", linewidth=0.5,
                       label=sp.capitalize(), zorder=3)

        acc = (model.predict(X2_sc) == y).mean()
        ax.set_xlabel(feat_labels[0], fontsize=9)
        ax.set_ylabel(feat_labels[1], fontsize=9)
        ax.set_title(f"{name}  (acc={acc:.3f})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_facecolor("#FAFAFA")

    axes.flat[5].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Decision Boundary Visualisation — Iris Classification\n")
    print("  Feature pair: petal length vs petal width")
    print("  Mesh resolution: 300×300\n")
    plot_decision_boundaries(
        os.path.join(RESULTS_DIR, "decision_boundaries.png"))
    print("\nDecision boundary visualisation complete.")


if __name__ == "__main__":
    main()
