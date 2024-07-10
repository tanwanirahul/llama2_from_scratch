'''
    Contains implementation of the Llama2 model components.
'''
import torch
from torch import nn as nn
from config import LlamaConfig
from cache import KVCache


class LlamaSelfAttention(nn.Module):
    '''
        Implements Llama SelfAttention mechanism.
    '''
    def __init__(self, config:LlamaConfig, layer_index:int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        assert config.n_embed % config.n_q_heads == 0 , f"cannot divide the totoal dimensions {config.n_embed} among {config.n_q_heads}."
        self.head_dims = config.n_embed // config.n_q_heads

        self.q_proj = nn.Linear(config.n_embed, config.n_q_heads * self.head_dims, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_kv_heads * self.head_dims, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_kv_heads * self.head_dims, bias=False)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        self._rotary_emb = LlamaRotaryEmbeddings(config)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        pass

class LlamaMLP(nn.Module):
    '''
        Implements MLP module of the decoder block.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.n_embed, config.interm_dims, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.interm_dims, bias=False)
        self.down_proj = nn.Linear(config.interm_dims, config.n_embed, bias=False)
        self.activation = nn.SiLU

    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        pass

class LlamaDecoderLayer(nn.Module):
    '''
        Implements the llama transformer decoder module.
    '''
    def __init__(self, config:LlamaConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
    
        self.self_attn = LlamaSelfAttention(config, layer_index=layer_index)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        pass

class LlamaRotaryEmbeddings(nn.Module):
    '''
        Rotary embedding model layer for the Llama model.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
    
    def forward(self, input_ids, attention_mask, device="cpu"):
        pass

class SwiGLU(nn.Module):
    '''
        Implements SwiGLU activation function.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
    
    def forward(self, input_ids, attention_mask, device="cpu"):
        pass

class LlamaRMSNorm(nn.Module):
    '''
        Implements RMSNorm as is used in Llama model implementation.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.ones(config.n_embed))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, input_ids, attention_mask, device="cpu"):
        pass

class Llama(nn.Module):
    '''
        Implements Llama2 model.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, lay_idx) for lay_idx in range(config.n_layers)
        ])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        pass
    
class LlamaWithLMHead(nn.Module):
    '''
        Implements Llama2 model along with the head linear module.
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.model = Llama(config)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        pass