# Designed and implemented Transformer model from scratch using NumPy, replicating 'Attention Is All You Need' architecture; ported to PyTorch for efficient GPU training on a synthetic dataset.

## Overview

This project implements the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) entirely from scratch using NumPy. Due to the computational requirements of training, the implementation was then ported to PyTorch to leverage GPU acceleration.

## Transformer Architecture

![Transformer Architecture](https://nlp.stanford.edu/~johnhew/img/transformer.png)

*Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)*

## Implementation Details

### NumPy Implementation

The initial implementation follows a modular approach, constructing the Transformer step-by-step with key components:

- **Embedding Layer**: Converts token indices into dense vector representations.
- **Positional Encoding**: Adds sinusoidal positional encodings to retain sequential information.
- **Multi-Head Attention**: Implements self-attention and masked self-attention (for autoregressive tasks).
- **Layer Normalization**: Normalizes activations for stable training.
- **Feed-Forward Neural Network**: Fully connected layers applied after attention.
- **Encoder Layer & Encoder**: Stacks multiple encoder layers.
- **Decoder Layer & Decoder**: Implements the decoder with masked self-attention and encoder-decoder attention.
- **Dropout Layer**: Regularization technique to prevent overfitting.
- **Transformer Model**: Combines all modules into the final Transformer architecture.

### PyTorch Implementation

Due to the high computational cost of training a Transformer model from scratch, the full implementation was ported to PyTorch, enabling:

- Efficient GPU-accelerated training.
- Autograd support for backpropagation.
- Seamless data loading and batch processing.

## Training Details

- A synthetic dataset was generated for testing the Transformer model.
- The PyTorch version was trained on this dataset to validate the implementation.
- Loss reduction and model outputs were observed to assess training effectiveness.

## Usage

### Running the NumPy Implementation

```bash
python numpy_transformer.py
```

This runs the Transformer model built using only NumPy.

### Running the PyTorch Implementation

```bash
python torch_transformer.py
```

This version utilizes PyTorch for GPU acceleration and model training.

## Results & Insights

- The PyTorch implementation demonstrated significantly faster training times due to GPU acceleration.
- The model effectively learned the synthetic dataset, confirming correctness.
- Comparison between NumPy and PyTorch highlighted the challenges of implementing deep learning architectures manually.

## Future Work

- Training on real-world NLP datasets such as WMT or IWSLT.
- Implementing optimizations like FlashAttention.
- Extending the model to support larger-scale tasks.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

