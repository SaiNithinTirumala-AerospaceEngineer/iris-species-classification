# Methodology — Iris Species Classification

## Overview

This project applies five supervised classification algorithms to the
UCI Iris dataset, comparing accuracy, precision, recall, F1, and ROC-AUC
across KNN, SVM, Decision Tree, Random Forest, and Logistic Regression.
Decision boundary visualisation provides intuitive insight into how each
classifier partitions the feature space.

---

## Dataset

UCI Iris dataset — Fisher (1936). 150 samples, 4 features, 3 classes
(50 samples each — perfectly balanced). Features measure sepal and petal
dimensions in centimetres.

### Feature separability

| Feature | Setosa range | Versicolor range | Virginica range | Separable? |
|---|---|---|---|---|
| Sepal length | 4.3–5.8 cm | 4.9–7.0 cm | 4.9–7.9 cm | Partial |
| Sepal width | 2.3–4.4 cm | 2.0–3.4 cm | 2.2–3.8 cm | Poor |
| Petal length | 1.0–1.9 cm | 3.0–5.1 cm | 4.5–6.9 cm | Strong |
| Petal width | 0.1–0.6 cm | 1.0–1.8 cm | 1.4–2.5 cm | Strong |

Petal features fully separate Iris setosa from the other two classes.
Versicolor and virginica overlap in petal length (3.0–5.1 vs 4.5–6.9).

---

## Models

### K-Nearest Neighbours (KNN)
Non-parametric — classifies by majority vote among k nearest neighbours
in feature space. Sensitive to feature scaling. Best k selected via
GridSearchCV from [3, 5, 7, 9, 11].

### Support Vector Machine (SVM)
Finds the maximum-margin hyperplane separating classes. RBF kernel maps
data to higher-dimensional space to handle non-linear boundaries.
Regularisation C and kernel width γ tuned via GridSearchCV.

### Decision Tree
Recursive binary splitting on feature thresholds. Fully interpretable
— produces a human-readable set of if-then rules. Prone to overfitting
without depth constraints.

### Random Forest
Ensemble of decision trees trained on bootstrap samples with random
feature subsets. Reduces variance vs single tree. Feature importances
derived from mean decrease in Gini impurity.

### Logistic Regression
Linear classifier using the softmax function for multi-class output.
Regularisation parameter C controls overfitting. Despite being a linear
model, achieves 100% test accuracy on this dataset — confirming the
strong linear separability of the Iris classes after feature scaling.

---

## Training strategy

- **Split:** 80/20 stratified (120 train, 30 test)
- **Scaling:** StandardScaler — zero mean, unit variance
- **Tuning:** 5-fold stratified GridSearchCV on training set
- **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC on test set

---

## Key findings

- Logistic Regression achieves **100% test accuracy** — the Iris
  classes are linearly separable after StandardScaler normalisation
- SVM and Random Forest achieve **96.7% test accuracy**
- Petal length and petal width are the dominant discriminating features
- Setosa is trivially separable — all classifiers achieve 100% recall
  on setosa; misclassifications occur only between versicolor/virginica
- Decision boundary plots show SVM produces the smoothest boundary —
  Decision Tree produces axis-aligned rectangular regions

---

## References

- Fisher, R.A. (1936) The use of multiple measurements in taxonomic
  problems. Annals of Eugenics, 7(2), 179–188.
- UCI Machine Learning Repository — Iris Dataset.
- Scikit-learn Documentation — https://scikit-learn.org