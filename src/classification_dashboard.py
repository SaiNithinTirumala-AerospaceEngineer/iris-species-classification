"""
classification_dashboard.py
---------------------------
Six-panel classification dashboard — the hero README figure.

Panels:
  1. CV accuracy comparison — all five models
  2. Test accuracy + F1 metrics table
  3. Decision boundary — best model (SVM)
  4. Confusion matrix — best model
  5. Feature importance — petal vs sepal separability
  6. Pipeline summary

Inputs : data/ (all processed arrays)
Outputs: results/classification_dashboard.png

Usage:
    python src/classification_dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#BA7517"]
SP_COLOURS  = ["#378ADD","#1D9E75","#D85A30"]
BG_COLOURS  = ["#D6E8F8","#C8EFE2","#F8DDD4"]


def load_and_rebuild():
    X_tr = np.load(os.path.join(DATA_DIR,"X_train.npy"))
    X_te = np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_tr = np.load(os.path.join(DATA_DIR,"y_train.npy"))
    y_te = np.load(os.path.join(DATA_DIR,"y_test.npy"))
    classes = np.load(os.path.join(DATA_DIR,"class_names.npy"),
                      allow_pickle=True)

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
            weights=s["KNN"]["params"].get("weights","uniform")),
        "SVM": SVC(kernel="rbf", probability=True,
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
    for m in models.values():
        m.fit(X_tr, y_tr)

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_te)
        acc = (y_pred == y_te).mean()
        results[name] = {
            "acc": acc,
            "cv":  s[name]["cv_mean"],
            "cv_std": s[name]["cv_std"],
            "y_pred": y_pred,
        }

    return models, results, X_tr, X_te, y_tr, y_te, classes


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating classification dashboard...")

    models, results, X_tr, X_te, y_tr, y_te, classes = load_and_rebuild()
    names = list(results.keys())

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Iris Species Classification Dashboard — Five Classifier Comparison\n"
        "Bharat Intern Machine Learning Internship  ·  Aug–Sep 2023  ·  "
        "UCI Iris Dataset (150 samples, 3 classes)",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.45)

    # ── Panel 1: CV accuracy ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,0])
    cv_means = [results[n]["cv"]     for n in names]
    cv_stds  = [results[n]["cv_std"] for n in names]
    bars = ax1.bar(range(len(names)), cv_means, color=COLOURS,
                   edgecolor="white", width=0.6)
    ax1.errorbar(range(len(names)), cv_means, yerr=cv_stds,
                 fmt="none", color="black", capsize=5, linewidth=1.2)
    ax1.bar_label(bars, fmt="%.4f", padding=3, fontsize=8.5,
                  fontweight="bold")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=7.5)
    ax1.set_ylabel("CV Accuracy (5-fold)", fontsize=10)
    ax1.set_title("Cross-Validation Accuracy", fontsize=11,
                  fontweight="bold")
    ax1.set_ylim(0.85, 1.08)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_facecolor("#FAFAFA")

    # ── Panel 2: Test accuracy table ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,1])
    ax2.axis("off")
    table_data = [[n, f"{results[n]['acc']:.4f}",
                   "✓" if results[n]['acc']>=0.97 else ""]
                  for n in names]
    tbl = ax2.table(cellText=table_data,
                    colLabels=["Model","Test Accuracy","≥97%"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)
    for (r,c), cell in tbl.get_celld().items():
        if r==0:
            cell.set_facecolor("#1D9E75")
            cell.set_text_props(color="white", fontweight="bold")
        elif r%2==0:
            cell.set_facecolor("#F0F9F5")
    ax2.set_title("Test Set Accuracy", fontsize=11,
                  fontweight="bold", pad=70)

    # ── Panel 3: Decision boundary — SVM ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0,2])
    df = pd.read_csv(os.path.join(ROOT,"data","iris.csv"))
    sp_map = {"setosa":0,"versicolor":1,"virginica":2}
    y_all  = df["species"].map(sp_map).values
    X2     = df[["petal_length_cm","petal_width_cm"]].values
    sc2    = StandardScaler(); X2_sc = sc2.fit_transform(X2)

    svm2 = SVC(kernel="rbf",
               C=float(json.load(open(os.path.join(
                   MODELS_DIR,"training_summary.json")))["SVM"]["params"]["C"]),
               random_state=RANDOM_SEED)
    svm2.fit(X2_sc, y_all)

    x_min,x_max = X2_sc[:,0].min()-0.4, X2_sc[:,0].max()+0.4
    y_min,y_max = X2_sc[:,1].min()-0.4, X2_sc[:,1].max()+0.4
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,200),
                        np.linspace(y_min,y_max,200))
    Z = svm2.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax3.contourf(xx,yy,Z, cmap=ListedColormap(BG_COLOURS), alpha=0.75)
    for i,(sp,col) in enumerate(zip(["setosa","versicolor","virginica"],SP_COLOURS)):
        mask = y_all==i
        ax3.scatter(X2_sc[mask,0],X2_sc[mask,1],color=col,s=30,
                    edgecolors="black",linewidth=0.4,
                    label=sp.capitalize(),zorder=3)
    ax3.set_xlabel("Petal Length (scaled)",fontsize=9)
    ax3.set_ylabel("Petal Width (scaled)",fontsize=9)
    ax3.set_title("SVM Decision Boundary\nPetal features",
                  fontsize=11,fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#FAFAFA")

    # ── Panel 4: Confusion matrix — best model ────────────────────────────
    ax4 = fig.add_subplot(gs[1,0])
    best = max(results, key=lambda k: results[k]["acc"])
    cm = confusion_matrix(y_te, results[best]["y_pred"])
    im = ax4.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax4, shrink=0.85)
    for i in range(3):
        for j in range(3):
            ax4.text(j,i,str(cm[i,j]),ha="center",va="center",
                     fontsize=13,fontweight="bold",
                     color="white" if cm[i,j]>cm.max()*0.6 else "black")
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels([c.capitalize() for c in classes],rotation=15,fontsize=9)
    ax4.set_yticklabels([c.capitalize() for c in classes],fontsize=9)
    ax4.set_xlabel("Predicted",fontsize=10)
    ax4.set_ylabel("Actual",fontsize=10)
    ax4.set_title(f"Confusion Matrix — {best}\n"
                  f"Test Accuracy={results[best]['acc']:.4f}",
                  fontsize=11,fontweight="bold")

    # ── Panel 5: Feature separability bar chart ───────────────────────────
    ax5 = fig.add_subplot(gs[1,1])
    feat_names = ["Sepal\nLength","Sepal\nWidth",
                  "Petal\nLength","Petal\nWidth"]
    df_raw = pd.read_csv(os.path.join(ROOT,"data","iris.csv"))
    feats  = ["sepal_length_cm","sepal_width_cm",
              "petal_length_cm","petal_width_cm"]
    sp_list = ["setosa","versicolor","virginica"]
    x = np.arange(len(feat_names))
    width = 0.25
    for i,(sp,col) in enumerate(zip(sp_list,SP_COLOURS)):
        means = [df_raw[df_raw.species==sp][f].mean() for f in feats]
        ax5.bar(x + i*width, means, width, label=sp.capitalize(),
                color=col, edgecolor="white")
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(feat_names, fontsize=9)
    ax5.set_ylabel("Mean value (cm)", fontsize=10)
    ax5.set_title("Feature Means by Species\n(separability overview)",
                  fontsize=11,fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(axis="y",alpha=0.25)
    ax5.set_facecolor("#FAFAFA")

    # ── Panel 6: Summary ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1,2])
    ax6.axis("off")
    metrics = [
        ("Dataset",          "UCI Iris — Fisher (1936)"),
        ("Samples",          "150 (120 train / 30 test)"),
        ("Features",         "4 (sepal + petal dims)"),
        ("Classes",          "3 (balanced, 50 each)"),
        ("Models compared",  "5"),
        ("Best model",       best),
        ("Best test acc",    f"{results[best]['acc']:.4f}"),
        ("CV strategy",      "5-fold stratified"),
        ("Tuning",           "GridSearchCV"),
        ("Scaling",          "StandardScaler"),
        ("Key finding",      "Petal dims > sepal dims"),
        ("Internship",       "Bharat Intern ML · 2023"),
    ]
    for i,(label,value) in enumerate(metrics):
        y_pos = 0.95 - i*0.077
        ax6.text(0.02,y_pos,label+":",transform=ax6.transAxes,
                 fontsize=9,color="#555555")
        ax6.text(0.48,y_pos,value,transform=ax6.transAxes,
                 fontsize=9,fontweight="bold",color="#1A1A1A")
    ax6.set_title("Pipeline Summary",fontsize=11,fontweight="bold")
    ax6.add_patch(plt.Rectangle((0,0),1,1,fill=False,
                                 edgecolor="#CCCCCC",linewidth=1,
                                 transform=ax6.transAxes))

    out = os.path.join(RESULTS_DIR,"classification_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print("\nClassification dashboard complete.")


if __name__ == "__main__":
    main()
