"""
BitNet a4.8 Implementation

A PyTorch implementation of BitNet a4.8 with 4-bit activations for 1-bit LLMs.
Includes hybrid quantization and sparsification strategy for efficient inference.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.nn.parameter import Parameter


@dataclass
class BitNetConfig:
    """Configuration class for BitNet a4.8 model hyperparameters."""
    
    hidden_size: int = 4096
    intermediate_size: int = 11008  # GLU size
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False


class RMSNorm(nn.Module):
    """RMSNorm implementation with optional bias."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Args:
            hidden_size: Dimensionality of input features
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = Parameter(torch.ones(hidden_size))
        self.eps = eps
        
        logger.debug(f"Initialized RMSNorm with hidden_size={hidden_size}, eps={eps}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class Int4Quantizer(nn.Module):
    """4-bit integer quantization using absmean scaling."""
    
    def __init__(self):
        super().__init__()
        self.min_val = -8
        self.max_val = 7
        
    def forward(self, x: Tensor) -> Tensor:
        """Quantize input tensor to 4-bit integers."""
        beta = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        scale = beta * math.sqrt(7)
        x_q = torch.round(x * math.sqrt(7) / (beta + 1e-5))
        x_q = torch.clamp(x_q, self.min_val, self.max_val)
        return x_q * scale / math.sqrt(7)


class Int8Quantizer(nn.Module):
    """8-bit integer quantization using absmax scaling."""
    
    def __init__(self):
        super().__init__()
        self.min_val = -128
        self.max_val = 127
        
    def forward(self, x: Tensor) -> Tensor:
        """Quantize input tensor to 8-bit integers."""
        gamma = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_q = torch.round(x * 127 / (gamma + 1e-5))
        x_q = torch.clamp(x_q, self.min_val, self.max_val)
        return x_q * gamma / 127


class TopKSparsifier(nn.Module):
    """Applies top-k sparsification to input tensor."""
    
    def __init__(self, k: float = 0.5):
        """
        Args:
            k: Fraction of values to keep (between 0 and 1)
        """
        super().__init__()
        self.k = k
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply top-k sparsification.
        
        Returns:
            Tuple of (sparse tensor, binary mask)
        """
        abs_x = torch.abs(x)
        num_keep = int(self.k * x.shape[-1])
        threshold = torch.kthvalue(abs_x, k=num_keep, dim=-1, keepdim=True)[0]
        mask = (abs_x >= threshold).float()
        return x * mask, mask


class BitLinear(nn.Module):
    """1.58-bit quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: Size of input features
            out_features: Size of output features
        """
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.scale = Parameter(torch.ones(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply quantized linear transformation."""
        # Quantize weights to {-1, 0, 1}
        w_abs = torch.abs(self.weight)
        w_mean = torch.mean(w_abs)
        w_q = torch.round(self.weight / (w_mean + 1e-5))
        w_q = torch.clamp(w_q, -1, 1)
        
        return F.linear(x, w_q * w_mean * self.scale)


class BitNetAttention(nn.Module):
    """Multi-head attention with 4-bit activation quantization."""
    
    def __init__(self, config: BitNetConfig):
        """
        Args:
            config: Model configuration object
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.k_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.v_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size)
        
        self.input_quant = Int4Quantizer()
        self.output_quant = Int8Quantizer()
        self.output_sparse = TopKSparsifier(k=0.5)
        
        logger.debug(f"Initialized BitNetAttention with {self.num_heads} heads")
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply multi-head attention with quantized activations.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Quantize input
        hidden_states = self.input_quant(hidden_states)
        
        # Project Q/K/V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states) 
        value = self.v_proj(hidden_states)
        
        # Reshape heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Output projection with quantization and sparsification
        context = self.output_quant(context)
        context, _ = self.output_sparse(context)
        output = self.o_proj(context)
        
        return output


class BitNetMLP(nn.Module):
    """FFN with gated linear unit and hybrid quantization."""
    
    def __init__(self, config: BitNetConfig):
        """
        Args:
            config: Model configuration object
        """
        super().__init__()
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size)
        
        self.input_quant = Int4Quantizer()
        self.down_quant = Int8Quantizer()
        
        logger.debug(f"Initialized BitNetMLP with intermediate_size={config.intermediate_size}")
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply FFN transformation with quantized activations."""
        # Input quantization
        x = self.input_quant(x)
        
        # Up projection with gating
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # ReLU^2 activation
        gate = F.relu(gate).pow(2)
        
        # Combine gate and up projections
        intermediate = up * gate
        
        # Down projection with 8-bit quantization
        intermediate = self.down_quant(intermediate)
        output = self.down_proj(intermediate)
        
        return output


class BitNetBlock(nn.Module):
    """Transformer block with BitNet a4.8 quantization."""
    
    def __init__(self, config: BitNetConfig):
        """
        Args:
            config: Model configuration object
        """
        super().__init__()
        self.attention = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply transformer block with quantized operations."""
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BitNetModel(nn.Module):
    """Complete BitNet a4.8 model."""
    
    def __init__(self, config: BitNetConfig):
        """
        Args:
            config: Model configuration object
        """
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BitNetBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        logger.info(f"Initialized BitNetModel with {config.num_hidden_layers} layers")
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states


def create_model(
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_hidden_layers: int = 32,
    num_attention_heads: int = 32,
    **kwargs,
) -> BitNetModel:
    """
    Create a BitNet a4.8 model with the specified configuration.
    
    Args:
        hidden_size: Model dimension
        intermediate_size: FFN intermediate dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        **kwargs: Additional config parameters
        
    Returns:
        Initialized BitNetModel
    """
    config = BitNetConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        **kwargs
    )
    
    model = BitNetModel(config)
    return model


# Example usage
if __name__ == "__main__":
    logger.info("Creating BitNet a4.8 model")
    
    model = create_model(
        hidden_size=816,  # 4096/5
        intermediate_size=2202,  # 11008/5 
        num_hidden_layers=6,  # 32/5 rounded down
        num_attention_heads=6  # 32/5 rounded down
    )
    
    # Generate sample input
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, 32000, (batch_size, seq_length))
    
    # Forward pass
    logger.info("Running forward pass")
    outputs = model(input_ids)
    logger.info(f"Output shape: {outputs.shape}")
