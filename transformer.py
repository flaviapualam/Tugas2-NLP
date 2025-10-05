"""
Implementasi Decoder-only Transformer menggunakan NumPy.
"""

import numpy as np
import warnings
from typing import Tuple, Optional

# Suppress numerical warnings untuk output yang clean
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fungsi softmax yang stabil secara numerik."""
    # Kurangi nilai maksimum untuk stabilitas numerik
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Potong nilai ekstrem untuk mencegah overflow
    x_clipped = np.clip(x_shifted, -500, 500)
    
    exp_x = np.exp(x_clipped)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # Hindari pembagian dengan nol
    sum_exp = np.maximum(sum_exp, 1e-15)
    
    return exp_x / sum_exp


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x: np.ndarray) -> np.ndarray:
    """Aktivasi ReLU."""
    return np.maximum(0, x)


class TokenEmbedding:
    """Layer embedding token."""
    
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Gunakan inisialisasi yang lebih konservatif untuk stabilitas
        std = 0.02  # Nilai kecil untuk mencegah overflow
        self.weight = np.random.normal(0, std, (vocab_size, d_model))
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.weight[token_ids]


class PositionalEncoding:
    """Positional encoding sinusoidal."""
    
    def __init__(self, d_model: int, max_seq_len: int = 512):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class ScaledDotProductAttention:
    """Scaled dot-product attention dengan causal masking."""
    
    def __init__(self, d_k: int):
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)
    
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 1
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_k = q.shape
        
        # Hitung skor attention dengan safe operations
        scores = np.clip(np.matmul(q, k.transpose(0, 2, 1)) * self.scale, -100, 100)
        
        # Periksa nilai tidak valid dan ganti dengan angka kecil
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-1e9)
        
        # Terapkan causal mask
        mask = self.create_causal_mask(seq_len)
        scores = np.where(mask, -1e9, scores)  # Gunakan -1e9 bukan -inf untuk stabilitas
        
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, v)
        
        return output


class MultiHeadAttention:
    """Mekanisme multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Gunakan inisialisasi yang lebih konservatif
        std = 0.02
        self.W_q = np.random.normal(0, std, (d_model, d_model))
        self.W_k = np.random.normal(0, std, (d_model, d_model))
        self.W_v = np.random.normal(0, std, (d_model, d_model))
        self.W_o = np.random.normal(0, std, (d_model, d_model))
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Safe matrix multiplication dengan clipping
        q = np.clip(np.matmul(x, self.W_q), -100, 100)
        k = np.clip(np.matmul(x, self.W_k), -100, 100)
        v = np.clip(np.matmul(x, self.W_v), -100, 100)
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        attention_outputs = []
        for i in range(self.num_heads):
            head_output = self.attention.forward(q[:, i], k[:, i], v[:, i])
            attention_outputs.append(head_output)
        
        attention_output = np.stack(attention_outputs, axis=1)
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        output = np.clip(np.matmul(attention_output, self.W_o), -100, 100)
        
        return output


class FeedForward:
    """Jaringan feed-forward 2-layer dengan ReLU."""
    
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gunakan inisialisasi yang lebih konservatif
        std = 0.02
        
        self.W1 = np.random.normal(0, std, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.normal(0, std, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = relu(np.clip(np.matmul(x, self.W1) + self.b1, -100, 100))
        output = np.clip(np.matmul(hidden, self.W2) + self.b2, -100, 100)
        return output


class DecoderBlock:
    """Blok decoder dengan self-attention dan feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.d_model = d_model
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        norm_x = layer_norm(x, self.norm1_gamma, self.norm1_beta)
        attention_output = self.self_attention.forward(norm_x)
        x = x + attention_output
        
        norm_x = layer_norm(x, self.norm2_gamma, self.norm2_beta)
        ff_output = self.feed_forward.forward(norm_x)
        x = x + ff_output
        
        return x


class Transformer:
    """
    Model Decoder-only Transformer dengan weight tying.
    
    Menggunakan weight tying antara input token embedding dan output projection layer
    untuk mengurangi parameter dan meningkatkan performa.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, 
                 d_ff: int, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.decoder_blocks = []
        for _ in range(num_layers):
            self.decoder_blocks.append(DecoderBlock(d_model, num_heads, d_ff))
        
        self.final_norm_gamma = np.ones(d_model)
        self.final_norm_beta = np.zeros(d_model)
        
        # Weight tying: Berbagi bobot antara input embedding dan output projection
        # Output projection menggunakan transpose dari bobot token embedding
        self.output_projection = self.token_embedding.weight.T
    
    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.token_embedding.forward(token_ids)
        x = self.positional_encoding.forward(x)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block.forward(x)
        
        x = layer_norm(x, self.final_norm_gamma, self.final_norm_beta)
        logits = np.clip(np.matmul(x, self.output_projection), -100, 100)
        
        last_token_logits = logits[:, -1, :]
        next_token_probs = softmax(last_token_logits, axis=-1)
        
        return logits, next_token_probs


def test_transformer():
    """Tes fungsionalitas inti transformer: shapes, masking, dan generation."""
    print("Testing Decoder-only Transformer")
    
    # Buat model
    vocab_size, d_model, num_heads, num_layers, d_ff = 10, 64, 4, 2, 128
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Input tes
    batch_size, seq_len = 2, 4
    token_ids = np.array([[1, 4, 2, 7], [3, 1, 5, 2]])
    
    # Forward pass
    logits, next_token_probs = model.forward(token_ids)
    
    # Verifikasi shape
    shapes_correct = (logits.shape == (batch_size, seq_len, vocab_size) and 
                     next_token_probs.shape == (batch_size, vocab_size))
    print(f"✓ Shape verification: {shapes_correct}")
    print(f"  Logits: {logits.shape}, Probs: {next_token_probs.shape}")
    
    # Verifikasi probabilitas
    prob_sums = np.sum(next_token_probs, axis=-1)
    print(f"✓ Probability sums: {np.allclose(prob_sums, 1.0)}")
    
    # Verifikasi causal masking
    attention = ScaledDotProductAttention(d_k=16)
    mask = attention.create_causal_mask(4)
    print(f"✓ Causal mask working: {np.sum(mask) == 6} ({np.sum(mask)} positions masked)")
    print("  Causal mask matrix (1 = masked positions):")
    print(f"  {mask.astype(int)}")
    print("  Upper triangle positions are masked to prevent future attention")
    
    # Verifikasi weight tying
    print("\n✓ Weight Tying Verification:")
    embedding_shape = model.token_embedding.weight.shape
    output_proj_shape = model.output_projection.shape
    print(f"  Token embedding shape: {embedding_shape}")
    print(f"  Output projection shape: {output_proj_shape}")
    
    # Cek apakah output projection adalah transpose dari embedding
    is_transposed = np.array_equal(model.output_projection, model.token_embedding.weight.T)
    print(f"  Weight sharing verified: {is_transposed}")
    print(f"  Formula: output_projection = token_embedding.weight.T")
    
    # Hitung parameter savings
    without_tying = embedding_shape[0] * embedding_shape[1] * 2  # embedding + output projection
    with_tying = embedding_shape[0] * embedding_shape[1]  # shared weights
    savings = without_tying - with_tying
    percentage = (savings / without_tying) * 100
    print(f"  Parameter efficiency: {with_tying} params (saved {savings}, {percentage:.1f}% reduction)")

if __name__ == "__main__":
    test_transformer()