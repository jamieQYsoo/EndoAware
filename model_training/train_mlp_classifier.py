from pathlib import Path

from sklearn.neural_network import MLPClassifier

try:
    from model_training.common import MODEL_DIR, RANDOM_STATE, evaluate_classification, save_model, split_data
except ModuleNotFoundError:
    from common import MODEL_DIR, RANDOM_STATE, evaluate_classification, save_model, split_data

OUT_FILE = MODEL_DIR / "endometriosis_mlp_classifier.pkl"


def main() -> None:
    X_train, X_test, y_train, y_test = split_data()

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    save_model(model, OUT_FILE)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, y_pred, y_prob)

    print(f"Saved: {Path(OUT_FILE)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
