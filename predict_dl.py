import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h = h_n[-1]
        logits = self.fc(h)
        return logits


def tokenize(text: str):
    return text.lower().split()


def encode_text(text: str, vocab: Dict[str, int], max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab.get("<unk>", 1)) for tok in tokens]
    pad_id = vocab.get("<pad>", 0)
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def load_model(model_path: Path) -> (nn.Module, Dict[str, int], int):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train with train_dl.py first.")

    saved = torch.load(model_path, map_location="cpu")
    vocab = saved["vocab"]
    max_len = saved["max_len"]
    embed_dim = saved["embed_dim"]
    hidden_dim = saved["hidden_dim"]

    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
    )
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, vocab, max_len


def main():
    parser = argparse.ArgumentParser(description="Predict spam/ham using deep learning model (PyTorch).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="spam_model_dl.pt",
        help="Path to the deep learning model file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        required=True,
        help="One or more texts to classify.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model, vocab, max_len = load_model(model_path)

    label_idx_to_name = {0: "ham", 1: "spam"}

    texts = args.text
    encoded = [encode_text(t, vocab, max_len) for t in texts]
    inputs = torch.tensor(encoded, dtype=torch.long)

    with torch.inference_mode():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).tolist()

    for t, p in zip(texts, preds):
        label = label_idx_to_name.get(p, "unknown").upper()
        print(f"[{label}] {t}")


if __name__ == "__main__":
    main()
