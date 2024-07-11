'''
    Configuration parameters for Llama2-7B model.
'''
from dataclasses import dataclass
import torch

class LlamaConfig:
    '''
        Holds configuration for the Llama2 model.
    '''
    vocab_size: int = 32000
    n_embed: int = 4096
    n_layers: int = 32
    n_q_heads: int = 32
    n_kv_heads: int = 32
    interm_dims: int = 11008
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta_base: int = 10000
    default_dtype: torch.dtype = torch.float16
