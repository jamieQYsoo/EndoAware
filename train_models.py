from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from model_training.common import MODEL_DIR, RANDOM_STATE, evaluate_classification, save_model, split_data


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, y_pred, y_prob)

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    return metrics


def print_summary_table(results):
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    rows = [
        [
            result["model"],
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['roc_auc']:.4f}",
        ]
        for result in results
    ]

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def format_row(row_values):
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(row_values))

    separator = "-+-".join("-" * width for width in col_widths)

    print("\nSummary Evaluation Table")
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


X_train, X_test, y_train, y_test = split_data()

models = {
    "MLP Classifier": (
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        str(MODEL_DIR / "endometriosis_mlp_classifier.pkl"),
    ),
    "SVM Classifier": (
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        str(MODEL_DIR / "endometriosis_svm_classifier.pkl"),
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        str(MODEL_DIR / "endometriosis_logistic_classifier.pkl"),
    ),
    "Random Forest Classifier": (
        RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE),
        str(MODEL_DIR / "endometriosis_randomforest_classifier.pkl"),
    ),
    "XGBoost Classifier": (
        XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        str(MODEL_DIR / "endometriosis_xgb_classifier.pkl"),
    ),
}

summary_results = []

for name, (model, out_file) in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    save_model(model, Path(out_file))
    print(f"  Saved: {Path(out_file)}")
    print("  Evaluating on test set...")
    metrics = evaluate(name, model, X_test, y_test)
    summary_results.append({"model": name, **metrics})

    # Show a sample of predicted probability outputs (0-1)
    sample_probs = model.predict_proba(X_test.head(5))[:, 1]
    print("  Example probabilities (first 5 test rows):")
    print("   ", [round(float(p), 4) for p in sample_probs])

print_summary_table(summary_results)

print("\nAll classification models trained and evaluated successfully.")