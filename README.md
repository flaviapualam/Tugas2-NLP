# Tugas 2 - Implementasi Arsitektur Transformer
**Nama:** Flavia Hidayriamraata Pualam  
**NIM:** 22/494376/TK/54219

## Implementasi Decoder-only Transformer

Implementasi arsitektur decoder-only Transformer (GPT-style) dari awal menggunakan NumPy, tanpa menggunakan library deep learning apa pun.

## Fitur

- Arsitektur decoder-only (GPT-style)
- Multi-head self-attention dengan causal masking
- Layer normalization (pre-norm)
- Weight tying antara embedding input dan proyeksi output
- Implementasi murni dengan NumPy (tanpa framework eksternal)
  
## Clone repository
```bash
git clone https://github.com/flaviapualam/Tugas2-NLP.git
cd Tugas2-NLP
```

## Persiapan

Pastikan NumPy sudah terpasang:

```bash
pip install numpy
```

## Cara Menjalankan

### 1. Menjalankan tes utama
```bash
python transformer.py
```

### 2. Menjalankan tes komponen
```bash
python demo.py
```

## Contoh Penggunaan dalam Kode

```python
from transformer import Transformer
import numpy as np

# Membuat model Transformer
model = Transformer(vocab_size=10, d_model=64, num_heads=4, num_layers=2, d_ff=128)

# Input berupa ID token
token_ids = np.array([[1, 4, 2, 7]])

# Forward pass
logits, next_token_probs = model.forward(token_ids)
```

## Struktur File

- `transformer.py`: Implementasi utama dengan seluruh komponen arsitektur
- `demo.py`: Pengujian komponen dan validasi bentuk tensor
- `README.md`: Dokumentasi penggunaan dan hasil uji sederhana

## Bukti Uji Sederhana

### 1. Tes Utama
```
Testing Decoder-only Transformer
✓ Shape verification: True
  Logits: (2, 4, 10), Probs: (2, 10)
✓ Probability sums: True
✓ Causal mask working: True (6 positions masked)
  Causal mask matrix (1 = masked positions):
  [[0 1 1 1]
   [0 0 1 1]
   [0 0 0 1]
   [0 0 0 0]]
  Upper triangle positions are masked to prevent future attention

✓ Weight Tying Verification:
  Token embedding shape: (10, 64)
  Output projection shape: (64, 10)
  Weight sharing verified: True
  Formula: output_projection = token_embedding.weight.T
  Parameter efficiency: 640 params (saved 640, 50.0% reduction)
```

### 2. Tes Komponen
```
COMPONENT TESTING
✓ Token Embedding: (2, 3) → (2, 3, 8)
✓ Positional Encoding: (2, 3, 8) → (2, 3, 8)
✓ Causal Masking: 6 future positions blocked
  Causal mask matrix (1 = masked positions):
  [[0 1 1 1]
   [0 0 1 1]
   [0 0 0 1]
   [0 0 0 0]]
  Upper triangle positions are masked to prevent future attention
✓ No future attention verified: 0.000000

SHAPE VERIFICATION
Model: vocab=15, d_model=24, heads=3, layers=2
✓ Input: (2, 5)
✓ Output logits: (2, 5, 15)
✓ Next token probs: (2, 15)
✓ All shapes correct: True
```

### 3. Detail Causal Masking
```
Causal mask (1 = masked positions):
[[0 1 1 1]
 [0 0 1 1]
 [0 0 0 1]
 [0 0 0 0]]

Masked positions count: 6 (triangle above diagonal)
Verified: No attention flows to future positions
```

### 4. Weight Tying (BONUS)
```
Token embedding weights shape: (vocab_size, d_model)
Output projection shape: (d_model, vocab_size)
Weight tying verified: output_projection = embedding.weight.T

Efisiensi Parameter:
- Bobot antara embedding input dan proyeksi output dibagikan
- Penghematan memori hingga 50% pada parameter embedding
- Stabilitas training meningkat melalui weight sharing
```

---
