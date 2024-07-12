'''
    Contains implementation of the Llama2 model components.
'''
from dataclasses import dataclass
import torch
from torch import nn as nn
from config import LlamaConfig
from cache import KVCache
from transformers.utils.hub import cached_file
from safetensors import torch as safetorch


@dataclass
class LlamaOutput:
    logits: torch.FloatTensor

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
        self.n_kv_groups = config.n_q_heads // config.n_kv_heads
        assert config.n_q_heads % config.n_kv_heads == 0, f"no. of query heads must be divisble by no. of kv heads."

        self.q_proj = nn.Linear(config.n_embed, config.n_q_heads * self.head_dims, bias=False, dtype=config.default_dtype)
        self.k_proj = nn.Linear(config.n_embed, config.n_kv_heads * self.head_dims, bias=False, dtype=config.default_dtype)
        self.v_proj = nn.Linear(config.n_embed, config.n_kv_heads * self.head_dims, bias=False, dtype=config.default_dtype)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False, dtype=config.default_dtype)

        self.rotary_emb = LlamaRotaryEmbeddings(config, head_dims=self.head_dims)
    
    def forward(self, hidden_state, attention_mask, position_ids, kv_cache:KVCache=None, device="cpu"):
        '''
            Forward pass for the llama self attention.
        '''
        batch_size, seq_len, _ = hidden_state.shape

        # create q, k, v tensors from the hidden state
        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        # convert tensors from [bs, seq_len, emb_dim] => [bs, n_heads, seq_len, head_dims]
        q = q.view(batch_size, seq_len, self.config.n_q_heads, self.head_dims).transpose(1,2)
        k = k.view(batch_size, seq_len, self.config.n_kv_heads, self.head_dims).transpose(1,2)
        v = v.view(batch_size, seq_len, self.config.n_kv_heads, self.head_dims).transpose(1,2)

        # add rotary embeddings to the query and keys
        cos_m_theta, sin_m_theta = self.rotary_emb(v, position_ids, v.device.type)
        q, k = self._add_rotary_embeddings(q, k, cos_m_theta, sin_m_theta)

        # update latest kv and get the past k,v from the cache.
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_index)
            
        assert k.shape[-2] == attention_mask.shape[-2], f"Seq len for keys and attention mask didn't match."

        # copy kv heads to match the no. of q/attn heads for pytorch's sdpa.        
        k = self._copy_kv_heads(k, copies=self.n_kv_groups)
        v = self._copy_kv_heads(v, copies=self.n_kv_groups)
        
        # sdpa doesn't work properly on cuda if the qkv are non contiguous.
        if q.device.type == "cuda":
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        # use torch's sdpa for faster attention implementation.
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask)
        
        #convert attn_output from [bs, n_q_heads, seq_len, head_dims] => [bs, seq_len, embd_dims]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.config.n_q_heads * self.head_dims)

        # run the attention output through a final linear projection layer.
        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def _add_rotary_embeddings(self, q, k, cos_m_theta, sin_m_theta):
        '''
            Adds rotary embeddings to q and k states.
        '''
        heads_dim = 1
        
        cos_m_theta = cos_m_theta.unsqueeze(heads_dim)
        sin_m_theta = sin_m_theta.unsqueeze(heads_dim)

        rotate_q = self._rotate_q_k_halfs(q)
        rotate_k = self._rotate_q_k_halfs(k)
        
        q = (q * cos_m_theta) + (rotate_q * sin_m_theta)
        k = (k * cos_m_theta) + (rotate_k * sin_m_theta)

        return q, k
    
    def _copy_kv_heads(self, korv, copies=1):
        '''
            if no. of kv_heads is less than no. q heads, this method
            creates copies of kv heads to match it with no. of q heads.
            No. of KV heads and No. of Q heads needs to be same for pytorch's
            scaled_dot_product_attention.
        '''
        if copies == 1:
            return korv

        bs, n_kv_heads, seq_len, head_dims = korv.shape
        korv = korv[:, :, None, :, :].expand(bs, n_kv_heads, copies, seq_len, head_dims)
        korv = korv.reshape(bs, n_kv_heads*copies, seq_len, head_dims)

        return korv
    
    def _rotate_q_k_halfs(self, qork):
        '''
            Rotates q or k tensors to compute [-q/2:, 0:q/2] or 
            [-k/2:, 0:k/2]. This is required for computing rotary positional embeddings.
        '''
        dims = qork.shape[-1]
        qork_upper = qork[:,:,:,:dims//2]
        qork_lower = qork[:,:,:, dims//2:]
        return torch.cat([-qork_lower, qork_upper], dim=-1)
        

class LlamaMLP(nn.Module):
    '''
        Implements MLP module of the decoder block.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.n_embed, config.interm_dims, bias=False, dtype=config.default_dtype)
        self.up_proj = nn.Linear(config.n_embed, config.interm_dims, bias=False, dtype=config.default_dtype)
        self.down_proj = nn.Linear(config.interm_dims, config.n_embed, bias=False, dtype=config.default_dtype)
        self.activation = nn.SiLU()

    def forward(self, hidden_state):
        '''
            Forward pass for the Llama MLP block of the decoder module:
            down(act(gate) * up)
        '''
        return self.down_proj(self.activation(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

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
    
    def forward(self, hidden_state, attention_mask, position_ids, kv_cache=None, device="cpu"):
        '''
            Implememts the forward pass for the llama decoder layer.
        '''
        residual = hidden_state
        
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask, position_ids, kv_cache, device)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)

        hidden_state = residual + hidden_state
        
        return hidden_state

class LlamaRotaryEmbeddings(nn.Module):
    '''
        Rotary embedding model layer for the Llama model.
    '''
    def __init__(self, config:LlamaConfig, head_dims:float):
        super().__init__()
        self.config = config
        base = config.rope_theta_base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dims, 2, dtype=torch.float16) / head_dims))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, value_state, position_ids, device_type="cpu"):
        '''
            Computes cos(mQ) and sin(mQ) as per the RoFormer paper.
            Paper -> https://arxiv.org/pdf/2104.09864v4.
            `hidden_state` is the input tensor with the shape [bs, n_attn_heads, seq_len, head_dims]
            `position_ids` are the positions of the input sequence of the shape [bs, seq_len]
        '''
        batch_size = position_ids.shape[0]

        # convert thetas from [head_dims/2] -> [bs, head_dims/2, 1]
        theta = self.inv_freq[None,:,None].float().expand(batch_size, -1, 1)

        # convert m/position_ids from [bs, seq_len] -> [bs, 1, seq_len]
        m = position_ids[:, None, :].float()

        # disable autocast / amp to force computation with float32 precision
        with torch.autocast(device_type=device_type, enabled=False):
            # computes m_theta of the shape [bs, seq_len, head_dims/2]
            m_theta = torch.matmul(theta.float(), m.float()).transpose(1, 2)
            # concat m_thetas to form the shape of [bs, seq_len, head_dims]
            m_theta = torch.cat([m_theta, m_theta], dim=-1)
            
            # compute cos and sin of mQ
            cos_m_theta = m_theta.cos()
            sin_m_theta = m_theta.sin()

        return cos_m_theta.to(value_state.dtype), sin_m_theta.to(value_state.dtype)

class LlamaRMSNorm(nn.Module):
    '''
        Implements RMSNorm as is used in Llama model implementation.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.ones(config.n_embed, dtype=config.default_dtype))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states):
        '''
            RMSNorm as used by Llama.
            ai = ai / rms * gi
            gi -> gain parameter
            rms(a) = sqrt(mean(ai^2)) + eps
        '''
        dtype = hidden_states.dtype
        g = self.weight

        # compute 1/rms(a)
        inv_rms = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) 
        
        # return a/rms(a) * g
        return (hidden_states * inv_rms * g)

class Llama(nn.Module):
    '''
        Implements Llama2 model.
    '''
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed, dtype=config.default_dtype)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, lay_idx) for lay_idx in range(config.n_layers)
        ])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        '''
            Forward pass for the Llama model.
        '''
        # create token embeddings of the shape: [bs, seq_len, n_embd]
        token_embds = self.embed_tokens(input_ids)

        # initialize kv_cache for the prefill phase.
        if kv_cache is None:
            kv_cache = KVCache()

        # Prepare position_ids.
        cached_tokens = kv_cache.seq_len(0)
        cached_adj_positions = torch.arange(cached_tokens, cached_tokens + token_embds.shape[1])
        # convert from [seq_len] to [1, seq_len]
        position_ids = cached_adj_positions.unsqueeze(0)

        #prepare attention / causal mask.
        causal_mask = self._prepare_causal_mask(token_embds, attention_mask, cached_adj_positions)
        
        # Now that we have the token embeddings, and the attention_mask prepared,
        # we're ready to proceed to the decoder layers.
        hidden_state = token_embds
        for layer in self.layers:
            hidden_state = layer(hidden_state, causal_mask, position_ids, kv_cache, device=input_ids.device)

        # post decoder/pre-linearhead layer norm.
        hidden_state = self.norm(hidden_state)

        return hidden_state

    def _prepare_causal_mask(self, input_ids, attention_mask, cached_adj_positions):
        '''
            Prepares attention mask for causal inference in the self attention layer.
            Attention mask needs to take into account all of the following:
            1. Previously processed/cached tokens that we do not want to run attention on needs to be masked.
            2. Padding tokens needs to be masked as well.
            3. Causal (upper right diagonal) mask for regressive text generation.
            
            `input_ids`: input data tensor of the shape [bs, seq_len]
            `attention_mask`: attention mask containing paddding information of the shape [bs, len(cached_token) + seq_len]
            `cached_adj_positions`: tensor containing the positions range taking into account the previously processed/cached positions.
            
            if the current seq_len is 4 (input ids of the shape [bs, 4]) and 2 tokens have been
            processed/cached previously, then the data for input parameters would loook like below.
            
            >> input_ids: [bs, 4]
            >> cahced_adj_positions: [2, 3, 4, 5]
            >> attention_mask: [bs, 6]
        '''
        batch_size = input_ids.shape[0]
        inp_dtype, inp_device = input_ids.dtype, input_ids.device

        min_value = torch.finfo(inp_dtype).min

        # Current Seq Length, and Total Target Length
        seq_length, target_length = input_ids.shape[1], attention_mask.shape[1]
        
        # create a causal mask with all the positions masked.
        causal_mask = torch.full((seq_length, target_length), fill_value=min_value, dtype=inp_dtype).to(device=inp_device)
        
        # incorporate seq positions and keep all the tokens above the positions masked.
        causal_mask *= torch.arange(target_length) > cached_adj_positions.reshape(-1, 1)
        
        # convert causal_mask from [seq_len, target_len] -> [bs, 1, seq_len, target_len]
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        # incorporate padding masks based on passed attention_mask
        if attention_mask is not None:
            assert causal_mask.shape[-1] == attention_mask.shape[-1], f"last dimension(target_len) for causal_mask and attention_mask must be same."
            padding_mask = (causal_mask + attention_mask[:, None, None, :]) == 0
            causal_mask = causal_mask.masked_fill(padding_mask, min_value)

        # torch's F.scaled_dot_product_attention cannot handle attention_mask rows that are completely
        # masked. It returns NaN on CUDA devices. To overcome, all such rows are completely
        # unmasked and the attention weights get evenly distributed.
        # Reference: https://github.com/pytorch/pytorch/issues/110213
        if causal_mask.device.type == "cuda":
            causal_mask.mul(~torch.all(causal_mask == min_value, dim=-1, keepdim=True))
        return causal_mask

class LlamaWithLMHead(nn.Module):
    '''
        Implements Llama2 model along with the head linear module.
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.model = Llama(config)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False, dtype=config.default_dtype)
    
    def forward(self, input_ids, attention_mask, kv_cache=None, device="cpu"):
        '''
            Forward pass for the Llama model.
        '''
        decoder_output = self.model(input_ids, attention_mask, kv_cache, device)
        logits = self.lm_head(decoder_output)
        return LlamaOutput(logits=logits)

    @classmethod
    def from_pretrained(cls, hf_model):
        '''
            Load the parameters for LlamaWithLMHead from the given 
            HF model.
        '''
        # create an instance of our Llama model.
        model = LlamaWithLMHead(LlamaConfig()).to(hf_model.dtype)

        # get the model's state / parameters.
        sd = model.state_dict()
        hf_sd = hf_model.state_dict()
        
        assert len(sd.keys()) == len(hf_sd.keys()), f"mismatch in model keys! expected: {len(sd.keys())}; found: {len(hf_sd.keys())}"

        # Copy the state from hf model to our model.
        for k in sd.keys():
            assert sd[k].shape == hf_sd[k].shape, f"Shape of the key: {k} didn't match!"
            assert sd[k].dtype == hf_sd[k].dtype, f"Type for the key: {k} didn't match!"

            with torch.no_grad():
                sd[k].copy_(hf_sd[k])
        return model
    
    @classmethod
    def from_checkpoint(cls, model_name, checkpoints, debug=False):
        '''
            Loads the model weights from the safetensor checkpoints. 
        '''
        model = LlamaWithLMHead(LlamaConfig())
        model = model.to(torch.float16)   
        sd = model.state_dict()
        sd_keys = sd.keys()

        safe_tensor_checkpoints = checkpoints
        for checkpoint in safe_tensor_checkpoints:
            if debug:
                print(f'loading checkpoint -> {checkpoint}')
            safe_tensors_file = cached_file(model_name, checkpoint)
            loaded_tensors = safetorch.load_file(safe_tensors_file)
            for key in loaded_tensors.keys():
                if "rotary_emb" in key and debug:                    
                    print(f"skipping {key}")
                else:
                    if debug:
                        print(f"loading {key} with the shape {loaded_tensors[key].shape}")
                    with torch.no_grad():
                        sd[key].copy_(loaded_tensors[key])    
        return model