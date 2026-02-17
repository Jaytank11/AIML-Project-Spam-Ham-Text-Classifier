import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import joblib
import torch
from torch import nn
from typing import Optional


class TextClassifier(nn.Module):
    """Deep learning model class (same as in train_dl.py)"""
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


class SpamHamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam/Ham Classifier - NLP Project")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")

        # Model variables
        self.classical_model = None
        self.dl_model = None
        self.dl_vocab = None
        self.dl_max_len = None
        self.model_type = tk.StringVar(value="classical")

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="📧 Spam/Ham Text Classifier",
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(pady=20)

        # Model selection frame
        model_frame = tk.Frame(self.root, bg="#f0f0f0")
        model_frame.pack(pady=10)

        tk.Label(
            model_frame,
            text="Select Model:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        ).pack(side=tk.LEFT, padx=10)

        classical_radio = tk.Radiobutton(
            model_frame,
            text="Classical (TF-IDF + Logistic Regression)",
            variable=self.model_type,
            value="classical",
            font=("Arial", 10),
            bg="#f0f0f0",
            command=self.on_model_change
        )
        classical_radio.pack(side=tk.LEFT, padx=5)

        dl_radio = tk.Radiobutton(
            model_frame,
            text="Deep Learning (LSTM)",
            variable=self.model_type,
            value="deep_learning",
            font=("Arial", 10),
            bg="#f0f0f0",
            command=self.on_model_change
        )
        dl_radio.pack(side=tk.LEFT, padx=5)

        # Input frame
        input_frame = tk.Frame(self.root, bg="#f0f0f0")
        input_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(
            input_frame,
            text="Enter your message:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        ).pack(anchor=tk.W)

        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=8,
            width=70,
            font=("Arial", 11),
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=2
        )
        self.text_input.pack(pady=10, fill=tk.BOTH, expand=True)
        self.text_input.insert("1.0", "Enter your message here...")
        self.text_input.bind("<FocusIn>", self.clear_placeholder)

        # Button frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        classify_btn = tk.Button(
            button_frame,
            text="🔍 Classify Message",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            cursor="hand2",
            command=self.classify_text
        )
        classify_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 12),
            bg="#95a5a6",
            fg="white",
            padx=15,
            pady=10,
            relief=tk.RAISED,
            cursor="hand2",
            command=self.clear_text
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = tk.Frame(self.root, bg="#f0f0f0")
        result_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(
            result_frame,
            text="Classification Result:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        ).pack(anchor=tk.W)

        self.result_label = tk.Label(
            result_frame,
            text="Waiting for input...",
            font=("Arial", 14),
            bg="#ecf0f1",
            fg="#2c3e50",
            relief=tk.SOLID,
            borderwidth=2,
            padx=20,
            pady=15,
            anchor=tk.CENTER
        )
        self.result_label.pack(pady=10, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 9),
            bg="#34495e",
            fg="white",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def clear_placeholder(self, event):
        if self.text_input.get("1.0", tk.END).strip() == "Enter your message here...":
            self.text_input.delete("1.0", tk.END)

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Waiting for input...", bg="#ecf0f1", fg="#2c3e50")
        self.status_label.config(text="Ready")

    def load_models(self):
        """Load both classical and deep learning models"""
        try:
            # Load classical model
            classical_path = Path("spam_model.joblib")
            if classical_path.exists():
                self.classical_model = joblib.load(classical_path)
                self.status_label.config(text="✓ Classical model loaded")
            else:
                self.status_label.config(text="⚠ Classical model not found. Train with train.py first.")

            # Load deep learning model
            dl_path = Path("spam_model_dl.pt")
            if dl_path.exists():
                saved = torch.load(dl_path, map_location="cpu")
                self.dl_vocab = saved["vocab"]
                self.dl_max_len = saved["max_len"]
                embed_dim = saved["embed_dim"]
                hidden_dim = saved["hidden_dim"]

                self.dl_model = TextClassifier(
                    vocab_size=len(self.dl_vocab),
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_classes=2
                )
                self.dl_model.load_state_dict(saved["model_state"])
                self.dl_model.eval()
                self.status_label.config(text="✓ Both models loaded successfully")
            else:
                if self.classical_model:
                    self.status_label.config(text="⚠ Deep learning model not found. Train with train_dl.py for DL option.")
                else:
                    self.status_label.config(text="⚠ No models found. Train models first.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.status_label.config(text="❌ Error loading models")

    def on_model_change(self):
        """Update status when model selection changes"""
        model_name = "Classical" if self.model_type.get() == "classical" else "Deep Learning"
        self.status_label.config(text=f"Using {model_name} model")

    def tokenize(self, text: str):
        """Simple tokenizer for deep learning model"""
        return text.lower().split()

    def encode_text(self, text: str, vocab: dict, max_len: int):
        """Encode text for deep learning model"""
        tokens = self.tokenize(text)
        ids = [vocab.get(tok, vocab.get("<unk>", 1)) for tok in tokens]
        pad_id = vocab.get("<pad>", 0)
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def classify_text(self):
        """Classify the input text"""
        text = self.text_input.get("1.0", tk.END).strip()

        if not text or text == "Enter your message here...":
            messagebox.showwarning("Warning", "Please enter a message to classify.")
            return

        try:
            model_type = self.model_type.get()

            if model_type == "classical":
                if self.classical_model is None:
                    messagebox.showerror("Error", "Classical model not loaded. Train with train.py first.")
                    return

                prediction = self.classical_model.predict([text])[0]
                label = prediction.upper()

            else:  # deep_learning
                if self.dl_model is None or self.dl_vocab is None:
                    messagebox.showerror("Error", "Deep learning model not loaded. Train with train_dl.py first.")
                    return

                encoded = self.encode_text(text, self.dl_vocab, self.dl_max_len)
                inputs = torch.tensor([encoded], dtype=torch.long)

                with torch.inference_mode():
                    outputs = self.dl_model(inputs)
                    pred_idx = outputs.argmax(dim=1).item()

                label_map = {0: "HAM", 1: "SPAM"}
                label = label_map.get(pred_idx, "UNKNOWN")

            # Update result display
            if label == "SPAM":
                bg_color = "#e74c3c"
                fg_color = "white"
                icon = "🚫"
            else:
                bg_color = "#27ae60"
                fg_color = "white"
                icon = "✓"

            self.result_label.config(
                text=f"{icon} {label}",
                bg=bg_color,
                fg=fg_color,
                font=("Arial", 16, "bold")
            )

            self.status_label.config(text=f"Classified as {label} using {model_type.replace('_', ' ').title()} model")

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_label.config(text="❌ Classification error")


def main():
    root = tk.Tk()
    app = SpamHamGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
