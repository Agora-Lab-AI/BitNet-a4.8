
# BitNet a4.8: 4-bit Activations for 1-bit LLMs

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Join Agora](https://img.shields.io/badge/Join-Agora-green.svg)](https://agoralab.xyz)

This repository contains an unofficial PyTorch implementation of [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965) (Wang et al., 2024).

## 📑 Paper Summary

BitNet a4.8 is a groundbreaking approach that enables 4-bit activations for 1-bit Large Language Models (LLMs). The method employs a hybrid quantization and sparsification strategy to mitigate quantization errors from outlier channels while maintaining model performance.

Key features:
- 4-bit quantization for attention and FFN inputs
- 8-bit quantization with sparsification for intermediate states
- Only 55% of parameters activated during inference
- Support for 3-bit KV cache
- Comparable performance to BitNet b1.58 with better inference efficiency

## 🚀 Implementation

This implementation includes:

```python
# Create a BitNet a4.8 model
model = create_model(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32
)
```

Key components:
- RMSNorm for layer normalization
- 4-bit and 8-bit quantizers
- TopK sparsification
- BitLinear (1.58-bit weights)
- Hybrid attention mechanism
- Gated FFN with ReLU²

## 📦 Installation

```bash
git clone https://github.com/yourusername/bitnet-a48
cd bitnet-a48
pip install -r requirements.txt
```

## 🤝 Join the Agora Community

This implementation is part of the [Agora](https://agoralab.xyz) initiative, where researchers and developers collaborate to implement cutting-edge ML papers. By joining Agora, you can:

- Collaborate with others on paper implementations
- Get early access to new research implementations
- Share your expertise and learn from others
- Contribute to open-source ML research

**[Join Agora Today](https://agoralab.xyz)**

## 📊 Results

The implementation achieves performance comparable to BitNet b1.58 while enabling:
- 4-bit activation compression
- 45% parameter sparsity
- Reduced inference costs
- 3-bit KV cache support

## 🛠️ Usage

```python
from bitnet_a48 import create_model

# Initialize model
model = create_model(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32
)

# Forward pass
outputs = model(input_ids, attention_mask)
```

## 📈 Training

The model uses a two-stage training recipe:
1. Train with 8-bit activations and ReLU²GLU
2. Fine-tune with hybrid quantization and sparsification

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Join the discussion on the [Agora Discord](https://agoralab.xyz/discord)!

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Original paper authors: Hongyu Wang, Shuming Ma, Furu Wei
- The Agora community
- PyTorch team
- Open-source ML community

## 📚 Citation

```bibtex
@article{wang2024bitnet,
  title={BitNet a4.8: 4-bit Activations for 1-bit LLMs},
  author={Wang, Hongyu and Ma, Shuming and Wei, Furu},
  journal={arXiv preprint arXiv:2411.04965},
  year={2024}
}
```

## 🔗 Links

- [Original Paper](https://arxiv.org/abs/2411.04965)
- [Agora Platform](https://agoralab.xyz)
- [Implementation Details](docs/IMPLEMENTATION.md)
- [Contributing Guide](docs/CONTRIBUTING.md)

Join us in implementing more cutting-edge ML research at [Agora](https://agoralab.xyz)!
