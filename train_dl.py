import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def tokenize(text: str) -> List[str]:
    # Simple whitespace + lowercase tokenizer
    return text.lower().split()


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # Reserve 0 for padding, 1 for unknown
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq, embed_dim)
        _, (h_n, _) = self.lstm(emb)
        h = h_n[-1]  # (batch, hidden_dim)
        logits = self.fc(h)
        return logits


def prepare_data(
    csv_path: Path, max_len: int = 50, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("CSV must contain 'label' and 'text' columns.")

    texts = df["text"].astype(str).tolist()
    labels_str = df["label"].astype(str).tolist()

    label_to_idx = {"ham": 0, "spam": 1}
    labels = [label_to_idx[l] for l in labels_str]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    vocab = build_vocab(X_train)

    X_train_ids = [encode_text(t, vocab, max_len) for t in X_train]
    X_test_ids = [encode_text(t, vocab, max_len) for t in X_test]

    train_ds = TextDataset(X_train_ids, y_train)
    test_ds = TextDataset(X_test_ids, y_test)
    return train_ds, test_ds, vocab


def train_epoch(
    model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train spam/ham classifier with a simple LSTM (PyTorch).")
    parser.add_argument("--data-path", type=str, default="spam_data.csv", help="Path to CSV dataset.")
    parser.add_argument("--model-path", type=str, default="spam_model_dl.pt", help="Path to save model state.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-len", type=int, default=50, help="Max sequence length.")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    train_ds, test_ds, vocab = prepare_data(
        csv_path=data_path,
        max_len=args.max_len,
        test_size=0.2,
        random_state=42,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - val_acc: {acc:.4f}")

    # Save model state and vocab/config
    save_data = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "max_len": args.max_len,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
    }
    torch.save(save_data, args.model_path)
    print(f"Saved deep learning model to {args.model_path}")


if __name__ == "__main__":
    main()
