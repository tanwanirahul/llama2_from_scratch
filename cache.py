'''
Implements KVCache required for Llama2.
'''
import torch
from typing import List

class KVCache:
    '''
        Keeps track of the keys and values computed at each layer.
        The Cache grows with every layer and with every iteration 
        through the self-attention.
    '''
    def __init__(self):
        '''
            Initializes an empty key and value cache.
            keys are tensors of the shape [B, n_heads, seq_len, head_dim]
            values are also tensors of the same shape [B, n_heads, seq_len, head_dim] 
        '''
        self.key_cache: List[torch.Tensor] = []
        self.val_cache: List[torch.Tensor] = []
        self._tokens_cached = 0 # Keeps track of how many tokens have been cached.
    
    def __len__(self):
        '''
            Returns length of the cache.
        '''
        return len(self.key_cache)

    def __iter__(self):
        '''
            Returns an iterator over kay,value cache.
        '''
        for layer_index in range(len(self)):
            yield (self.key_cache[layer_index], self.val_cache[layer_index])

    def __item__(self, layer_index:int):
        '''
            Returns Key/Value cached for the given layer index.
        '''
        if layer_index < len(self):
            return self.key_cache[layer_index], self.val_cache[layer_index]
        
        raise KeyError(f"Data for layer index {layer_index} isn't cached!")
    
    def update(self, key_states, value_states, layer_index):
        '''
            Updates the keys and values cache for the given layer
            `layer_index`. Returns the updated key,value tensors.
        '''
        if layer_index == 0:
            self._tokens_cached += key_states.shape[-2]
        
        if len(self) <= layer_index:
            self.key_cache.append(key_states)
            self.val_cache.append(value_states)
        else:
            self.key_cache[layer_index] = torch.cat([self.key_cache[layer_index], key_states], dim=-2)
            self.val_cache[layer_index] = torch.cat([self.key_cache[layer_index], key_states], dim=-2)
    
        return (self.key_cache[layer_index], self.val_cache[layer_index])

    def seq_len(self, layer_index):
        '''
            Returns the seq length for the given layer index.
        '''
        if len(self) <= layer_index:
            return 0
        # Shape of key_cache tensors -> [B, n_heads, seq_len, head_dims]
        return self.key_cache[layer_index].shape[-2]