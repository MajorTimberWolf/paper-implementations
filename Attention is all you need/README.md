# Transformer Implementation (Attention is All You Need)

This is an implementation of the Transformer model architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The implementation focuses on the decoder part of the transformer, similar to GPT models.

## Credits
This implementation builds upon the base attention code from [@cneuralnetwork](https://github.com/cneuralnetwork), with additional improvements including support for both learned and sinusoidal positional encodings.

## Model Architecture

### Core Components

1. **SelfAttentionHead**
   - Implements single-head self-attention mechanism
   - Components: Key, Query, Value projections
   - Uses causal masking for autoregressive generation
   - Includes dropout for regularization

2. **ParallelMultiHeadAttention**
   - Implements multi-head attention
   - Parallelizes multiple attention heads
   - Includes final projection layer and dropout
   - Concatenates outputs from all heads

3. **MLPBlock**
   - Feed-forward network following attention layer
   - Uses expansion factor of 4 for hidden layer
   - Includes ReLU activation and dropout
   - Processes token-wise features

4. **TransformerBlock**
   - Complete transformer block combining attention and MLP
   - Uses layer normalization before each sub-component (Pre-LN architecture)
   - Implements residual connections
   - Processes sequence of tokens maintaining temporal relationships

5. **GPTLanguageModel**
   - Main model class implementing the full architecture
   - Features:
     - Token embeddings
     - Position embeddings (both learned and sinusoidal options)
     - Multiple transformer blocks
     - Final layer normalization
     - Token prediction head
   - Includes text generation capabilities with temperature and top-k sampling

### Key Features

- **Dual Positional Encoding Options**:
  - Learned positional embeddings (default)
  - Sinusoidal positional encodings (as in original paper)
  
- **Training Features**:
  - Learning rate scheduling with warmup
  - AdamW optimizer
  - Gradient clipping
  - Train/validation loss tracking

- **Generation Features**:
  - Temperature-controlled sampling
  - Top-k sampling support
  - Autoregressive text generation

## Recent Fixes

### Attention Scaling Correction
- **Issue**: Previously, attention weights were incorrectly scaled by the full embedding dimension (`n_embd**-0.5`)
- **Fix**: Now correctly scales by the head dimension (`head_size**-0.5`) as per the original paper
- **Impact**: More accurate attention score computation, potentially leading to better model performance

Example of the difference:
- Old scaling (with n_embd=64): 0.125 (1/8)
- New scaling (with head_size=16): 0.25 (1/4)

This brings the implementation more in line with the original paper's specification.

## Usage

```python
# Initialize model
model = GPTLanguageModel(use_sinusoidal_pos=False)  # or True for sinusoidal encodings

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

# Generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(
    context, 
    max_new_tokens=2000, 
    temperature=0.8, 
    top_k=40
)[0].tolist())
```

## Model Configuration

- Batch Size: 16
- Block Size (Sequence Length): 32
- Embedding Dimension: 64
- Number of Heads: 4
- Number of Layers: 4
- Dropout: 0.0
- Learning Rate: 1e-3
- Training Iterations: 5000
- Evaluation Interval: 100

## References

1. Original Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Base Implementation: [@cneuralnetwork](https://github.com/cneuralnetwork) 