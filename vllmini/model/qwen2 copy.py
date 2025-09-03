import torch
from torch import nn
from transformers import Qwen2Config
from typing import List, Optional, Tuple
from paged_attention_cuda import paged_attention_v1, cache_ops

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # 使用标准Linear替代QuantLinear
        self.q_proj = nn.Linear(
            self.hidden_size,   
            self.num_attention_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )

        self.rope_theta = config.rope_theta
        self.rotary_dim = self.head_dim

        self.block_size = 16
        self.max_blocks = 1024

    def _rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        seq_len = q.shape[2]

        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32) / self.rotary_dim))
        inv_freq = inv_freq.unsqueeze(0)

        position_ids = position_ids.unsqueeze(-1).float()
        freqs = torch.matmul(position_ids, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        q_rot = q * freqs.cos() + self._rotate_half(q) * freqs.sin()
        k_rot = k * freqs.cos() + self._rotate_half(k) * freqs.sin()
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)
        q, k = self._rotary_pos_emb(q, k, position_ids)

        if use_cache:
            self._cache_kv(k, v, key_cache, value_cache, slot_mapping)

        if is_prefill:
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
        else:
            attn_output = self._paged_attention(q, key_cache, value_cache, block_table, seq_lens, max_seq_len)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return (attn_output, (k, v)) if use_cache else (attn_output, None)

    def _vanilla_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if self.num_attention_heads != self.num_key_value_heads:
            repeat_times = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat_times, dim=1)
            v = v.repeat_interleave(repeat_times, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    def _paged_attention(self, q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, 
                         block_table: torch.Tensor, seq_lens: torch.Tensor, max_seq_len: int):
        num_seqs, num_heads, seq_len, head_dim = q.shape
        q_flat = q.view(-1, num_heads, head_dim)
        out_flat = torch.empty_like(q_flat)

        paged_attention_v1(
            out_flat,
            q_flat,
            key_cache,
            value_cache,
            self.num_attention_heads,
            self.scale,
            block_table,
            seq_lens,
            self.block_size,
            max_seq_len,
            None,
            "auto",
            1.0,
            0,
            0,
            1,
            1,
            0,
        )
        return out_flat.view(num_seqs, num_heads, seq_len, head_dim)

    def _cache_kv(self, k: torch.Tensor, v: torch.Tensor, key_cache: torch.Tensor, 
                  value_cache: torch.Tensor, slot_mapping: torch.Tensor):
        batch_size, num_kv_heads, seq_len, head_dim = k.shape
        k_cache = k.transpose(1, 2).contiguous().view(-1, num_kv_heads, head_dim)
        v_cache = v.transpose(1, 2).contiguous().view(-1, num_kv_heads, head_dim)
        
        cache_ops.reshape_and_cache(
            k_cache,
            v_cache,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            1.0,
        )

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 使用标准Linear替代QuantLinear
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        return self.down_proj(intermediate)

class Qwen2Block(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.self_attn = Qwen2Attention(config)

        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            position_ids=position_ids,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return (hidden_states, attn_outputs[1]) if use_cache else hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id if config.pad_token_id is not None else -1
        )
        print(config.num_hidden_layers)
        config.num_hidden_layers = 24
        self.layers = nn.ModuleList([Qwen2Block(config) for _ in range(config.num_hidden_layers)])
        
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[List[torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        presents = () if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_slot_mapping = slot_mappings[i] if (slot_mappings and i < len(slot_mappings)) else None
            layer_block_table = block_tables[i] if (block_tables and i < len(block_tables)) else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=layer_slot_mapping,
                block_table=layer_block_table,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                position_ids=position_ids,
            )

            if use_cache:
                hidden_states, present = layer_outputs
                presents = presents + (present,)
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return (hidden_states, presents) if use_cache else (hidden_states,)

class Qwen2LMHeadModel(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[List[torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        model_outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states = model_outputs[0]

        logits = self.lm_head(hidden_states)

        outputs = logits
        if use_cache:
            return outputs,model_outputs[1]
        else: outputs,None

    def load_huggingface_weights(self, model_path: str):
        """加载HuggingFace格式的权重，忽略量化相关参数"""
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        hf_state_dict = hf_model.state_dict()

        # 过滤掉量化相关的权重键
        filtered_state_dict = {}
        for k, v in hf_state_dict.items():
            # 跳过GPTQ量化相关的参数
            if not any(part in k for part in ['qweight', 'qzeros', 'scales', 'g_idx']):
                filtered_state_dict[k] = v
        print("📝 Loading Qwen2 weights from HuggingFace model")
        # 加载词嵌入权重
        self._load_embed_tokens(filtered_state_dict)
        print("loading lm_head")
        # 加载lm_head权重
        self._load_lm_head(filtered_state_dict)
        print("loading norm")
        # 加载最终归一化层
        self._load_norm(filtered_state_dict)
        # 逐层加载Transformer层
        self._load_layers(filtered_state_dict)

        print(f"✅ Successfully loaded Qwen2 weights from {model_path} (ignored quantization parameters)")
    
    def _load_embed_tokens(self, hf_state_dict: dict):
        embed_key = "model.embed_tokens.weight"
        if embed_key in hf_state_dict:
            self.model.embed_tokens.load_state_dict({
                "weight": hf_state_dict[embed_key]
            })
        else:
            raise KeyError(f"Missing embed token key: {embed_key} in hf state_dict")

    def _load_lm_head(self, hf_state_dict: dict):
        lm_head_key = "lm_head.weight"
        if lm_head_key in hf_state_dict and not self.config.tie_word_embeddings:
            self.lm_head.load_state_dict({
                "weight": hf_state_dict[lm_head_key]
            })

    def _load_norm(self, hf_state_dict: dict):
        norm_key = "model.norm.weight"
        if norm_key in hf_state_dict:
            self.model.norm.load_state_dict({
                "weight": hf_state_dict[norm_key]
            })

    def _load_layers(self, hf_state_dict: dict):
        for layer_idx, layer in enumerate(self.model.layers):
            layer_prefix = f"model.layers.{layer_idx}."

            # 加载注意力层
            attn_prefix = layer_prefix + "self_attn."
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj_key = attn_prefix + proj_name + ".weight"
                if proj_key in hf_state_dict:
                    getattr(layer.self_attn, proj_name).load_state_dict({
                        "weight": hf_state_dict[proj_key]
                    })

            # 加载层归一化
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                norm_key = layer_prefix + norm_name + ".weight"
                if norm_key in hf_state_dict:
                    getattr(layer, norm_name).load_state_dict({
                        "weight": hf_state_dict[norm_key]
                    })

            # 加载MLP层
            mlp_prefix = layer_prefix + "mlp."
            for mlp_proj_name in ["gate_proj", "up_proj", "down_proj"]:
                mlp_proj_key = mlp_prefix + mlp_proj_name + ".weight"
                if mlp_proj_key in hf_state_dict:
                    getattr(layer.mlp, mlp_proj_name).load_state_dict({
                        "weight": hf_state_dict[mlp_proj_key]
                    })
