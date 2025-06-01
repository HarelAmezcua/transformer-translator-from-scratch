# 🌐 Neural Translator: Transformer from Scratch

A fully custom-built **neural machine translator** using the **Transformer architecture**, developed from the ground up in Python with PyTorch. This project explores the inner mechanics of attention, positional encoding, and sequence modeling—bringing theory to life in code.

---

## 📂 Project Structure

```

neural-translator/
├── data/                 # Dataset files (e.g., translation pairs)
├── models/               # Transformer model components (Encoder, Decoder, etc.)
├── utils/                # Utility functions (tokenization, data loaders, etc.)
├── train.py              # Training loop and model optimization
├── evaluate.py           # Evaluation metrics and inference
├── config.py             # Hyperparameters and config definitions
└── main.py               # Entry point for training and testing

````

---

## 🧪 Technologies Used

- **Python** — Core language
- **PyTorch** — Deep learning framework
- **matplotlib** — Visualization of training progress and attention maps
- **NumPy**, **tqdm**, etc. — Basic utilities

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/neural-translator.git
cd neural-translator

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training
python main.py --mode train

# 5. Translate a sentence
python main.py --mode translate --sentence "Hello, world!"
````

---

## 🧠 Core Features

* Custom **multi-head self-attention**
* LayerNorm, residual connections, and masking
* Positional encoding using sine and cosine
* Fully batched training with teacher forcing
* BLEU score evaluation and inference interface
* Attention visualization (optional)

---

## ✨ Example Usage

```python
from models.transformer import Transformer
from utils.tokenizer import tokenize_pair

# Sample input
src_sentence = "Hello, world!"
src_tensor = tokenize_pair(src_sentence)

# Load pre-trained model
model = Transformer.load_from_checkpoint("checkpoints/best_model.pt")
model.eval()

# Translate
output = model.translate(src_tensor)
print("Translation:", output)
```

---

## 📈 Training Visualization

> Example of training loss over epochs:

![Training Loss](assets/loss_curve.png)

> Attention heatmap:

![Attention Map](assets/attention_map.png)

---

## 📚 References

* Vaswani et al. (2017) — ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* PyTorch documentation and examples

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
