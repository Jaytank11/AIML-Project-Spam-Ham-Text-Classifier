import argparse
from pathlib import Path
from typing import List

import joblib


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first with train.py.")
    return joblib.load(model_path)


def predict_texts(model, texts: List[str]) -> List[str]:
    return model.predict(texts).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spam/Ham predictions using a trained model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="spam_model.joblib",
        help="Path to the trained model pipeline file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="*",
        help="One or more texts to classify.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a text file with one message per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    model = load_model(model_path)

    texts: List[str] = []

    if args.text:
        texts.extend(args.text)

    if args.input_file:
        input_file_path = Path(args.input_file)
        if not input_file_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_file_path}")
        with input_file_path.open("r", encoding="utf-8") as f:
            file_texts = [line.strip() for line in f if line.strip()]
        texts.extend(file_texts)

    if not texts:
        raise ValueError("No input text provided. Use --text or --input-file.")

    preds = predict_texts(model, texts)

    for t, p in zip(texts, preds):
        print(f"[{p.upper()}] {t}")


if __name__ == "__main__":
    main()
