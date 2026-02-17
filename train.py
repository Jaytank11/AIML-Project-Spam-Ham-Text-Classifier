import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    expected_cols = {"label", "text"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {expected_cols}, got {list(df.columns)}")
    return df


def build_pipeline() -> Pipeline:
    """
    Build a simple text classification pipeline:
    - TF-IDF vectorization
    - Logistic Regression classifier
    """
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=None,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate(data_path: Path, model_path: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

    print("\nPerformance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Saving trained model pipeline to: {model_path}")
    joblib.dump(pipeline, model_path)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Spam/Ham text classification model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="spam_data.csv",
        help="Path to CSV dataset with 'label' and 'text' columns.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="spam_model.joblib",
        help="Path to save the trained model pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)

    train_and_evaluate(
        data_path=data_path,
        model_path=model_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
