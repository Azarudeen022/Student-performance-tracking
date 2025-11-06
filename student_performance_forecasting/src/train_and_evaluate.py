#!/usr/bin/env python3
import argparse, json, os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, RocCurveDisplay)

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features

def evaluate(name, model, X_test, y_test, outdir):
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Fail","Pass"])
    plt.yticks(tick_marks, ["Fail","Pass"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=160)
    plt.close()

    # ROC curve plot (if proba available)
    if y_prob is not None:
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"{name} - ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=160)
        plt.close()

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--target", required=True, help="Target column name (binary 0/1)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--model_out", default="models/best_model.joblib")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Define models
    logreg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train and CV
    models = {"LogisticRegression": logreg, "RandomForest": rf}
    results = {}
    best_name, best_score, best_model = None, -np.inf, None

    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        metrics = evaluate(name, model, X_test, y_test, args.outdir)
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
        results[name] = metrics

        score = metrics.get("roc_auc", 0.0)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    # Save best model and metrics
    joblib.dump(best_model, args.model_out)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # If best is RF, dump simple feature importances (after one-hot) by approximating via RF feature_importances_
    if best_name == "RandomForest":
        # Extract the trained RF and one-hot feature names
        pre = best_model.named_steps["preprocess"]
        clf = best_model.named_steps["clf"]
        # Build feature names for exported transformer
        num_features = pre.transformers_[0][2]
        ohe = pre.transformers_[1][1].named_steps["onehot"]
        cat_features = pre.transformers_[1][2]
        cat_names = ohe.get_feature_names_out(cat_features)
        all_names = np.concatenate([num_features, cat_names])
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({"feature": all_names, "importance": importances})
            imp_df.sort_values("importance", ascending=False).to_csv(
                os.path.join(args.outdir, "feature_importances.csv"), index=False
            )

    print("Training complete.")
    print(f"Best model: {best_name} (ROC-AUC: {best_score:.3f})")
    print("Metrics saved to:", os.path.join(args.outdir, "metrics.json"))
    print("Model saved to:", args.model_out)

if __name__ == "__main__":
    main()
