from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model_training"
DATA_FILE = DATA_DIR / "cleaned_endometriosis_dataset.xlsx"

LABEL_COL = "diagnosis_label"
DROP_COLS = ["row"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATA_FILE, engine="openpyxl")


def split_data(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    df = load_dataset()
    X = df.drop(columns=[LABEL_COL] + DROP_COLS, errors="ignore")
    y = df[LABEL_COL].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def evaluate_classification(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_model(model, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_file)
