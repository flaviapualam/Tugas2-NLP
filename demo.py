"""
Skript demo untuk testing Decoder-only Transformer.
"""

import numpy as np
from transformer import Transformer, ScaledDotProductAttention, softmax


def detailed_component_test():
    """Tes komponen inti transformer."""
    print("COMPONENT TESTING")
    
    # Tes Token Embedding
    from transformer import TokenEmbedding
    embedding = TokenEmbedding(vocab_size=10, d_model=8)
    token_ids = np.array([[1, 4, 2], [3, 1, 5]])
    embeddings = embedding.forward(token_ids)
    print(f"✓ Token Embedding: {token_ids.shape} → {embeddings.shape}")
    
    # Tes Positional Encoding
    from transformer import PositionalEncoding
    pos_enc = PositionalEncoding(d_model=8, max_seq_len=10)
    x = np.random.randn(2, 3, 8)
    x_with_pos = pos_enc.forward(x)
    print(f"✓ Positional Encoding: {x.shape} → {x_with_pos.shape}")
    
    # Tes Causal Masking
    attention = ScaledDotProductAttention(d_k=4)
    mask = attention.create_causal_mask(4)
    print(f"✓ Causal Masking: {mask.sum()} future positions blocked")
    print("  Causal mask matrix (1 = masked positions):")
    print(f"  {mask.astype(int)}")
    print("  Upper triangle positions are masked to prevent future attention")
    
    # Verifikasi masking bekerja
    q = k = v = np.random.randn(1, 4, 4)
    scores = np.matmul(q, k.transpose(0, 2, 1)) / 2.0
    masked_scores = np.where(mask, -np.inf, scores)
    attention_weights = softmax(masked_scores, axis=-1)
    
    # Cek tidak ada future attention
    future_attention_total = 0
    for i in range(4):
        future_attention_total += np.sum(attention_weights[0, i, i+1:])
    print(f"✓ No future attention verified: {future_attention_total:.6f}")
    print()


def shape_verification():
    """Verifikasi tensor shapes melalui forward pass."""
    print("SHAPE VERIFICATION")
    
    # Buat model
    vocab_size, d_model, num_heads, num_layers, d_ff = 15, 24, 3, 2, 48
    batch_size, seq_len = 2, 5
    
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Model: vocab={vocab_size}, d_model={d_model}, heads={num_heads}, layers={num_layers}")
    
    # Forward pass
    logits, next_probs = model.forward(token_ids)
    
    # Verifikasi kebenaran
    expected_logits = (batch_size, seq_len, vocab_size)
    expected_probs = (batch_size, vocab_size)
    shapes_correct = (logits.shape == expected_logits and next_probs.shape == expected_probs)
    
    print(f"✓ Input: {token_ids.shape}")
    print(f"✓ Output logits: {logits.shape}")
    print(f"✓ Next token probs: {next_probs.shape}")
    print(f"✓ All shapes correct: {shapes_correct}")
    print()


if __name__ == "__main__":
    # Jalankan tes inti
    detailed_component_test()
    shape_verification()