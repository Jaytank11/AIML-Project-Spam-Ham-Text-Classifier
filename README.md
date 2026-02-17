## Spam/Ham Classification (NLP)

A complete end-to-end project for classifying SMS/email text as **spam** or **ham** using:
- **Classical ML**: TF-IDF + Logistic Regression
- **Deep Learning**: LSTM with PyTorch
- **GUI Application**: Tkinter-based desktop app

---

### 1. Environment Setup

From `d:\project`:

```bash
python -m pip install -r requirements.txt
```

**Note**: Tkinter (for GUI) comes built-in with Python on Windows. PyTorch will be installed automatically.

---

### 2. Project Structure

- `requirements.txt` – Python dependencies (numpy, pandas, scikit-learn, torch, etc.)
- `README.md` – this file
- `spam_data.csv` – **Expanded dataset** with 60+ examples (`label,text`)
- `train.py` – trains classical TF-IDF + Logistic Regression model
- `train_dl.py` – trains deep learning LSTM model (PyTorch)
- `predict.py` – command-line predictions using classical model
- `predict_dl.py` – command-line predictions using deep learning model
- `gui.py` – **GUI application** with both models

---

### 3. Dataset

The dataset (`spam_data.csv`) contains **60+ labeled examples**:
- **Ham messages**: Normal, legitimate messages (30+ examples)
- **Spam messages**: Promotional, scam, phishing messages (30+ examples)

Format: CSV with columns `label` and `text`.

You can replace it with your own larger dataset as long as the columns match.

---

### 4. Training Models

#### Classical Model (TF-IDF + Logistic Regression)

```bash
python train.py --data-path spam_data.csv --model-path spam_model.joblib
```

This will:
- Load the dataset
- Split into train/test (80/20)
- Train TF-IDF vectorizer + Logistic Regression
- Print accuracy, precision, recall, F1-score
- Save model to `spam_model.joblib`

#### Deep Learning Model (LSTM)

```bash
python train_dl.py --data-path spam_data.csv --model-path spam_model_dl.pt --epochs 15
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--max-len`: Maximum sequence length (default: 50)
- `--embed-dim`: Embedding dimension (default: 64)
- `--hidden-dim`: LSTM hidden dimension (default: 64)

---

### 5. GUI Application 🖥️

**Launch the GUI:**

```bash
python gui.py
```

**Features:**
- ✅ Clean, modern interface
- ✅ Switch between Classical and Deep Learning models
- ✅ Real-time classification
- ✅ Color-coded results (Green for HAM, Red for SPAM)
- ✅ Large text input area
- ✅ Status bar showing model loading status

**How to use:**
1. Run `python gui.py`
2. Select your preferred model (Classical or Deep Learning)
3. Type or paste your message in the text box
4. Click "🔍 Classify Message"
5. See the result displayed with color coding

**Note**: Make sure you've trained at least one model (`train.py` or `train_dl.py`) before using the GUI.

---

### 6. Command-Line Predictions

#### Using Classical Model

```bash
python predict.py --model-path spam_model.joblib --text "Congratulations, you won a free prize!!!"
```

Multiple texts:
```bash
python predict.py --model-path spam_model.joblib --text "hello how are you?" "win a free iPhone now!!!"
```

#### Using Deep Learning Model

```bash
python predict_dl.py --model-path spam_model_dl.pt --text "Congratulations, you won a free prize!!!"
```

---

### 7. Using Your Own Dataset

1. Prepare a CSV with columns: `label,text`
2. Put it in the project folder (e.g. `my_spam_dataset.csv`)
3. Train models:

```bash
python train.py --data-path my_spam_dataset.csv --model-path my_spam_model.joblib
python train_dl.py --data-path my_spam_dataset.csv --model-path my_spam_model_dl.pt
```

4. Use the models with `predict.py`, `predict_dl.py`, or `gui.py`

---

### 8. Model Comparison

| Feature | Classical Model | Deep Learning Model |
|---------|---------------|---------------------|
| **Speed** | Very Fast | Slower (CPU) |
| **Accuracy** | Good with enough data | Better with large datasets |
| **Training Time** | Seconds | Minutes (on CPU) |
| **Model Size** | Small (~MB) | Larger (~MB) |
| **Best For** | Quick prototyping, small datasets | Large datasets, better accuracy |

---

### 9. Notes

- The **expanded dataset** (`spam_data.csv`) now has 60+ examples for better training.
- Both models are saved separately and can be used independently.
- The GUI automatically detects which models are available.
- For production use, consider:
  - Larger, real-world datasets
  - Hyperparameter tuning
  - Cross-validation
  - Model evaluation on held-out test sets

---

### 10. Troubleshooting

**GUI won't start:**
- Make sure Python has Tkinter (usually built-in on Windows)
- Check that models are trained (`spam_model.joblib` or `spam_model_dl.pt` exist)

**ModuleNotFoundError:**
- Run `python -m pip install -r requirements.txt` again
- Make sure you're using the same Python interpreter everywhere

**Low accuracy:**
- The current dataset is small (60 examples). For better results, use a larger real-world dataset.
- Try training the deep learning model with more epochs: `--epochs 30`

---

**Enjoy classifying spam! 🎉**
