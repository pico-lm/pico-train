"""
The Pico Model: A Lightweight Transformer Language Model

Pico uses a simple LLAMA-style transformer architecture, written for clarity and educational purposes.

Everything is written with a modular design for easy modification and experimentation.

Key features:
- RMSNorm for layer normalization
- Rotary Positional Embeddings (RoPE)
- Multi-head attention with KV-cache support
- SwiGLU activation function
- Residual connections throughout

- KV-cache for faster autoregressive generation

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - LLAMA: https://arxiv.org/abs/2302.13971

Adapted from:
    - OLMO: https://github.com/allenai/OLMo
    - LLAMA: https://github.com/meta/llama
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from dataclasses import asdict

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutput

# typing imports
from typing import Union, Tuple, Optional, TYPE_CHECKING, Dict, Any

try:
    if TYPE_CHECKING:
        # We need to do this to avoid importing these when creating the HF-compatible models
        from src.config import ModelConfig
        import lightning as L
except ImportError:
    pass

########################################################
#
# Layer Normalization
#
########################################################


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    A variant of Layer Normalization that uses RMS statistics instead of mean/variance,
    resulting in improved stability and performance.

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Configuration object containing normalization parameters
            - config.norm_eps: Small constant for numerical stability
            - config.d_model: Model dimension for the weight parameter

    References:
        https://arxiv.org/abs/1910.07467
    """

    def __init__(self, config: Union["ModelConfig", "PicoHFConfig"]):
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor by its RMS value.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor and scales it by the weight parameter.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


########################################################
#
# Positional Embedding
#
########################################################


class RoPE(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Implements position-dependent rotation of keys and queries in attention mechanism,
    allowing better modeling of relative positions in sequences. Uses complex number
    operations for efficient rotation.

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Model configuration containing:
            - config.position_emb_theta: Base for frequency computation
            - config.d_model: Model dimension
            - config.attention_n_heads: Number of attention heads
            - config.max_seq_len: Maximum sequence length
        fabric (L.Fabric): Lightning Fabric instance for device management

    References:
        https://arxiv.org/abs/2104.09864
    """

    _freqs_cis: torch.Tensor = None

    def __init__(
        self, config: Union["ModelConfig", "PicoHFConfig"], fabric: "L.Fabric" = None
    ):
        super().__init__()

        self.fabric = fabric

        self.theta = config.position_emb_theta
        self.dim = config.d_model // config.attention_n_heads

        max_seq_len = config.max_seq_len

        # only gets set once, and then reused for all RoPE instances
        if RoPE._freqs_cis is None:
            RoPE._freqs_cis = self._setup_freqs_cis(max_seq_len, self.theta, self.dim)
            if fabric is not None:
                RoPE._freqs_cis = fabric.to_device(RoPE._freqs_cis)

    @classmethod
    def _setup_freqs_cis(cls, seq_len: int, theta: float, dim: int) -> torch.Tensor:
        """Setup Frequency Tensor for RoPE Embeddings

        Initializes the complex frequency tensor that is used to compute the RoPE embeddings.

        Note other implementations will use cos and sin directly, but using the complex
        number representation is (probably?) more efficient:

            e^(theta * i * t) = cos(theta * t) + i * sin(theta * t) [Euler's formula]
        """
        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, _freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    @torch.no_grad()
    def get_freqs_cis(
        self, input_shape: torch.Size, start_pos: int, end_pos: int
    ) -> torch.Tensor:
        """Reshape Frequency Tensor for RoPE Embeddings

        Makes the frequency tensor broadcastable with the input tensor.
        """
        _freqs_cis = RoPE._freqs_cis[start_pos:end_pos]
        ndim = len(input_shape)
        assert 0 <= 1 < ndim
        assert _freqs_cis.shape == (input_shape[1], input_shape[-1])

        # TODO: Check whether this is correct (might be able to remove this)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(input_shape)]
        return _freqs_cis.view(*shape)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        start_pos: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE Embeddings to Queries and Keys

        Applies the rotary positional embeddings to the input tensors via complex num multiplication

        NOTE: The start_pos is used if we want to use the kv_cache in the attention mechanism.
        """
        queries_ = torch.view_as_complex(
            queries.float().reshape(*queries.shape[:-1], -1, 2)
        )
        keys_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))

        input_shape = (
            queries_.shape
        )  # same as keys: (batch_size, seq_len, n_heads, head_dim/2)
        freqs_start_pos = start_pos
        freqs_end_pos = freqs_start_pos + queries_.shape[1]

        freqs_cis = self.get_freqs_cis(input_shape, freqs_start_pos, freqs_end_pos)
        # if fabric is set, freqs_cis is already on the correct device
        # otherwise, we need to move it to the correct device
        if self.fabric is not None:
            freqs_cis = self.fabric.to_device(freqs_cis)
        else:
            freqs_cis = freqs_cis.to(queries.device)

        queries_rotated = torch.view_as_real(queries_ * freqs_cis).flatten(3)
        keys_rotated = torch.view_as_real(keys_ * freqs_cis).flatten(3)
        return queries_rotated.type_as(queries), keys_rotated.type_as(keys)


########################################################
#
# Attention
#
########################################################


class Attention(nn.Module):
    """Multi-head Attention with Group Query Attention support.

    Implements scaled dot-product attention and supports:
    - Grouped Query Attention (GQA)
    - Key-Value caching for efficient inference
    - RoPE integration

    Args:
        config (Union[ModelConfig, PretrainedConfig]): Configuration containing:
            - config.attention_n_heads: Number of attention heads
            - config.attention_n_kv_heads: Number of key/value heads
            - config.d_model: Model dimension
            - config.batch_size: Maximum batch size
            - config.max_seq_len: Maximum sequence length
        fabric (L.Fabric): Lightning Fabric instance

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        config: Union["ModelConfig", "PicoHFConfig"],
        fabric: Optional["L.Fabric"] = None,
    ):
        super().__init__()

        self.fabric = fabric

        self.n_heads = config.attention_n_heads
        self.n_kv_heads = config.attention_n_kv_heads

        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len

        d_model = config.d_model
        self.head_dim = d_model // self.n_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=False)

        self.rope = RoPE(config, fabric)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for the attention mechanism.

        Computes queries, keys, and values for the attention mechanism. Applies rotary positional
        embeddings to the queries and keys, and then computes attention scores and outputs.

        For an introduction to the attention mechanism, see:
        https://arxiv.org/abs/1706.03762

        A few things to note:
        - The past_key_values is used to implement the KV cache, which is used to speed up
          generation by caching the KV pairs from previous forward passes. This is useful when doing
          tasks that require generating multiple tokens conditioned on previous tokens (e.g. language
          modeling, text generation, etc.). The way the KV cache is implemented is that each layer has
          its own KV cache - this KV cache is implemented as a tuple.
        """
        bsz, seq_len, _ = input.shape
        _queries, _keys, _values = (
            self.q_proj(input),
            self.k_proj(input),
            self.v_proj(input),
        )

        # Reshaping for multi-head attention
        queries = _queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = _keys.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values = _values.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # The start position is used to apply the RoPE embeddings to only the new tokens
        # when using the kv_cache in the attention mechanism.
        # We want to start from the last position in the cache.
        start_pos = past_key_values[0].shape[1] if past_key_values is not None else 0

        # apply rotary positional embeddings
        queries, keys = self.rope(queries, keys, start_pos)

        if past_key_values is not None:
            keys = torch.cat([past_key_values[0], keys], dim=1)
            values = torch.cat([past_key_values[1], values], dim=1)

        if use_cache:
            cached_keys = keys
            cached_values = values
        else:
            cached_keys = None
            cached_values = None

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # see if cuda available
        if self.fabric and self.fabric.device.type == "cuda":
            backend = SDPBackend.CUDNN_ATTENTION
        else:
            backend = SDPBackend.MATH

        with sdpa_kernel(backends=[backend]):
            attn_output = F.scaled_dot_product_attention(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                attn_mask=mask,
                enable_gqa=True if self.n_rep > 1 else False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, (cached_keys, cached_values)


########################################################
#
# SwiGLU (Combines MLP and Activation)
#
########################################################


class SwiGLU(nn.Module):
    """SwiGLU Activation Function with Linear Projections.

    Implements the SwiGLU activation function combined with linear transformations,
    serving as the feed-forward network in transformer blocks.

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Configuration containing:
            - config.d_model: Model dimension
            - config.activation_hidden_dim: Hidden dimension (typically 4 * d_model)

    References:
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: Union["ModelConfig", "PicoHFConfig"]):
        super().__init__()

        model_dim = config.d_model
        act_hidden_dim = config.activation_hidden_dim  # usually 4 * d_model

        self.w_0 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_1 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_2 = nn.Linear(act_hidden_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


########################################################
#
# PicoBlock and the Pico Model
#
########################################################


class PicoBlock(nn.Module):
    """Single Transformer Block with Attention and Feed-forward layers.

    Implements a standard transformer block with:
    - Multi-head attention with normalization and residual connection
    - SwiGLU feed-forward network with normalization and residual connection

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Model configuration; either a dataclass or
            a HuggingFace PicoHFConfig
        fabric (L.Fabric): Lightning Fabric instance
    """

    def __init__(
        self,
        config: Union["ModelConfig", "PicoHFConfig"],
        fabric: Optional["L.Fabric"] = None,
    ):
        super().__init__()

        self.attention = Attention(config, fabric)
        self.swiglu = SwiGLU(config)
        self.attention_norm = RMSNorm(config)
        self.swiglu_norm = RMSNorm(config)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attention_output, cached_key_values = self.attention(
            self.attention_norm(input),
            mask=mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        # NOTE: cached_key_values is None if use_cache is False

        h = input + attention_output
        out = h + self.swiglu(self.swiglu_norm(h))
        return out, cached_key_values


########################################################
#
# Pico Model
#
########################################################


class Pico(nn.Module):
    """
    Core Pico model: combines the embedding, Pico layers, and output projection into a single model.

    For more information on the model, see the classes for the modules that make up the model.
    """

    def __init__(
        self,
        model_config: Union["ModelConfig", "PicoHFConfig"],
        fabric: Optional["L.Fabric"] = None,
    ):
        super().__init__()
        self.config = model_config
        self.fabric = fabric

        self.embedding_proj = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList(
            [PicoBlock(self.config, self.fabric) for _ in range(self.config.n_layers)]
        )
        self.output_norm = RMSNorm(self.config)
        self.de_embedding_proj = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )

    def convert_to_hf_model(self) -> "PicoHF":
        """Convert the Lightning model to a HuggingFace model."""
        # Create HF config without fabric-specific settings
        hf_config = PicoHFConfig.from_dataclass(self.config)

        # Create new HF model
        hf_model = PicoHF(hf_config)

        # Copy state dict, excluding fabric-specific keys
        hf_model.load_state_dict(self.state_dict(prefix="pico."))

        return hf_model

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        This is the forward pass for the entire Pico model. It boils down to:
        - Embedding the input ids
        - Creating a causal mask
        - Processing through the pico layers
        - Projecting the output to logits

        NOTE: One feature that might be confusing is the KV cache. The KV cache is used to speed up
        generation by caching the KV pairs from previous forward passes. This is useful when doing
        tasks that require generating multiple tokens conditioned on previous tokens (e.g. language
        modeling, text generation, etc.). The way the KV cache is implemented is that each layer has
        its own KV cache which is stored as a tuple. The whole model then stores a tuple of these
        KV caches (so a tuple of tuples).
        """

        seq_len = input_ids.shape[-1]
        h = self.embedding_proj(input_ids)

        # Calculate start position from past cached KV pairs. Remember that each layer has its
        # own KV Cache. So when we index past_key_values, we need to index into the KV pairs for the
        # correct layer and then for either the keys or values.
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]

        # Create causal mask for current sequence
        mask = None
        if seq_len > 1:
            if self.fabric is not None:
                mask = self.fabric.to_device(
                    torch.full((seq_len, seq_len), float("-inf"))
                )
            else:
                mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)

            # If using KV cache, extend mask to cover cached sequence length
            if past_key_values is not None:
                # Add zeros for cached tokens (we can attend to all of them)
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask]).type_as(
                    h
                )

        # NOTE: If we are using the cache, we need to store the cached KV pairs for each layer
        #       in a tuple. Each layer will have its own cached KV pair which we aggregate in a tuple.
        cached_key_values = () if use_cache else None

        # Process through transformer blocks
        for idx, layer in enumerate(self.layers):
            layer_past_key_values = (
                past_key_values[idx] if past_key_values is not None else None
            )

            h, layer_cached_key_values = layer(
                h, mask=mask, past_key_values=layer_past_key_values, use_cache=use_cache
            )

            if use_cache:
                cached_key_values += (layer_cached_key_values,)

        # Final norm and projection
        h = self.output_norm(h)
        logits = self.de_embedding_proj(h).float()

        return logits, cached_key_values


########################################################
#
# PicoConfig and PicoForHF
#
########################################################

"""
HuggingFace wrapper for the Pico model.

Wait why do we need a wrapper? Aren't we just using the Pico class directly? Good question!

Many evaluation frameworks require a model be setup as a HuggingFace model, so we provide a simple
wrapper that does just that. When we save checkpoints of the Pico model, we save both the normal
Pico model as well as the model wrapped in this HuggingFace class.

This also lets you do cool things like: 

`model = AutoModelForCausalLM.from_pretrained("path/to/checkpoint")`
"""


class PicoHFConfig(PretrainedConfig):
    """HuggingFace config for Pico model."""

    model_type = "pico"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PicoHFConfig":
        # NOTE The typical from_dict method doesn't actually set the attributes unless they are
        # defined in the constructor.

        pico_config = cls(**kwargs)

        # Because this class is just a wrapper around the ModelConfig dataclass, we need to do
        # a little extra work to ensure that the attributes are actually set.
        for key, value in config_dict.items():
            setattr(pico_config, key, value)

        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        unused_kwargs = {
            key: value for key, value in kwargs.items() if not hasattr(pico_config, key)
        }

        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config

    @classmethod
    def from_dataclass(cls, model_config: "ModelConfig"):
        return cls.from_dict(asdict(model_config))


class PicoHF(PreTrainedModel):
    """HuggingFace wrapper for Pico model."""

    config_class = PicoHFConfig
    _no_split_modules = ["PicoBlock", "Attention", "SwiGLU", "RMSNorm"]

    def __init__(self, config: PicoHFConfig):
        super().__init__(config)
        self.pico = Pico(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[CausalLMOutput, CausalLMOutputWithPast]:
        """HuggingFace forward pass wrapper.

        Forwards pass for the HuggingFace version of the Pico Model. Basic wrapper around the
        Pico model's forward pass, and returns the output as a HuggingFace CausalLMOutput.
        """
        logits, past_key_values = self.pico(input_ids, past_key_values, use_cache)
        if use_cache:
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
            )
        else:
            return CausalLMOutput(
                logits=logits,
            )


# Register for auto classes
PicoHFConfig.register_for_auto_class()
PicoHF.register_for_auto_class("AutoModel")
PicoHF.register_for_auto_class("AutoModelForCausalLM")
