"""Train the hate speech classifier and save deployable artifacts.

Usage:
    python train.py --data "labeled dataset.csv"

Outputs (in ./artifacts):
    vectorizer.pkl, model.pkl, metrics.json, confusion_matrix.png
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from preprocess import clean_data

LABELS = {0: "Hate Speech", 1: "Offensive Language", 2: "No hate or offensive language"}
SEED = 42


def main(data_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df[["tweet", "class"]].dropna()
    df["label"] = df["class"].map(LABELS)
    print(f"Loaded {len(df)} rows\n{df['label'].value_counts()}\n")

    print("Cleaning text...")
    X_text = df["tweet"].apply(clean_data)
    y = df["label"].values

    # Stratified split keeps the (imbalanced) class ratios identical in train/test.
    Xtr_text, Xte_text, ytr, yte = train_test_split(
        X_text, y, test_size=0.2, random_state=SEED, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_text)   # fit on train ONLY (no leakage)
    Xte = vec.transform(Xte_text)

    # ---- Baseline (your original model) for comparison ----
    base = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    base_pred = base.predict(Xte)
    base_f1 = f1_score(yte, base_pred, average="macro")
    print(f"Baseline DecisionTree  acc={accuracy_score(yte, base_pred):.4f}  macro-F1={base_f1:.4f}")

    # ---- Improved model: balanced Logistic Regression, tuned on macro-F1 ----
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=2000),
        param_grid={"C": [0.5, 1, 2, 5, 10]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(Xtr, ytr)
    model = grid.best_estimator_
    pred = model.predict(Xte)
    macro_f1 = f1_score(yte, pred, average="macro")
    acc = accuracy_score(yte, pred)
    print(f"Best C = {grid.best_params_['C']}")
    print(f"LogReg (balanced)      acc={acc:.4f}  macro-F1={macro_f1:.4f}\n")

    report_txt = classification_report(yte, pred)
    print(report_txt)
    report = classification_report(yte, pred, output_dict=True)

    labels_sorted = sorted(LABELS.values())
    cm = confusion_matrix(yte, pred, labels=labels_sorted)
    print("Confusion matrix (rows=true, cols=pred):")
    print(labels_sorted)
    print(cm)

    # Save confusion matrix image for the README / LinkedIn
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["Hate Speech", "Offensive Language", "No hate or offensive language"]
        idx = [labels_sorted.index(n) for n in order]
        cm_plot = cm[np.ix_(idx, idx)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm_plot, cmap="YlGnBu")
        ticks = ["Hate", "Offensive", "Neither"]
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks)
        ax.set_yticks(range(3)); ax.set_yticklabels(ticks)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm_plot[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed; skipping confusion matrix image.")

    joblib.dump(vec, os.path.join(out_dir, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "classes": list(model.classes_),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "baseline_macro_f1": base_f1,
                "best_C": grid.best_params_["C"],
                "report": report,
            },
            f,
            indent=2,
        )
    print(f"\nSaved artifacts to ./{out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled dataset.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    main(args.data, args.out)
