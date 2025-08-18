import torch
from torch import nn
from transformers import Qwen2Config
from transformers import AutoConfig  
from typing import List, Optional, Tuple
from paged_attention_cuda import paged_attention_v1, cache_ops
import math


class QwenAttention(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim **-0.5

        # Qwen的注意力投影层（与GPT2的c_attn不同，Qwen通常分拆为q/k/v三个投影）
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        
        self.max_position_embeddings = config.max_position_embeddings
        self.block_size = 16  # 与项目KV缓存块大小保持一致
        self._init_rope()

    # def _init_rope(self):
    #     """初始化RoPE位置编码"""
    #     inv_freq = 1.0 / (self.rope_theta** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
    #     self.register_buffer("inv_freq", inv_freq, persistent=False)

    # def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    #     """对Q/K应用RoPE编码"""
    #     batch_size, seq_len, num_heads, head_dim = x.shape
    #     device = x.device

    #     # 计算位置编码
    #     inv_freq_expanded = self.inv_freq[None, :, None].expand(batch_size, -1, seq_len)  # [batch, head_dim/2, seq_len]
    #     position_ids_expanded = position_ids[:, None, :].float()  # [batch, 1, seq_len]
    #     freqs = torch.matmul(inv_freq_expanded, position_ids_expanded)  # [batch, head_dim/2, seq_len]
    #     emb = torch.cat([freqs.sin(), freqs.cos()], dim=1)  # [batch, head_dim, seq_len]
    #     emb = emb.transpose(1, 2).unsqueeze(2)  # [batch, seq_len, 1, head_dim]

    #     # 应用旋转编码
    #     x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)  # 拆分维度用于旋转
    #     x_rotated = torch.stack(
    #         [-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1
    #     ).reshape(batch_size, seq_len, num_heads, head_dim)  # 旋转操作
    #     x = x * emb[..., :head_dim] + x_rotated * emb[..., head_dim:]  # 融合位置信息
    #     return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # 投影Q/K/V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # # 应用RoPE位置编码（Qwen必需步骤）
        # q = self._apply_rope(q, position_ids)
        # k = self._apply_rope(k, position_ids)

        # 调整形状以适配缓存内核 [batch_size * seq_len, num_heads, head_dim]
        q = q.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
        v = v.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

        # 缓存K/V（使用项目的cache_ops内核）
        self._cache_kv(k, v, key_cache, value_cache, slot_mapping)

        if is_prefill:
            # 预填充阶段：使用常规注意力
            q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
            k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        else:
            # 解码阶段：使用Paged Attention内核
            attn_output = self._paged_attention(q, key_cache, value_cache, block_table, seq_lens, max_seq_len)
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return (attn_output, (k, v)) if use_cache else (attn_output, None)

    def _vanilla_attention(self, q, k, v, attention_mask):
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    def _cache_kv(self, k: torch.Tensor, v: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor):
        """使用项目的reshape_and_cache内核缓存K/V"""
        cache_ops.reshape_and_cache(
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            1.0,
        )

    def _paged_attention(self, q, key_cache, value_cache, block_table, seq_lens, max_seq_len):
        """调用Paged Attention内核进行高效解码"""
        num_seqs, num_heads, head_dim = q.shape
        out = torch.empty_like(q)
        paged_attention_v1(
            out,
            q,
            key_cache,
            value_cache,
            self.num_heads,
            self.scale,
            block_table,
            seq_lens,
            self.block_size,
            max_seq_len,
            None,  # alibi_slopes（Qwen不使用）
            "auto",
            1.0,
            0,
            0,
            1,
            1,
            0,
        )
        return out


class QwenMLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()  # Qwen使用SiLU激活

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class QwenBlock(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = QwenAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QwenMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        outputs = (hidden_states,) + attn_outputs[1:] if use_cache else (hidden_states,)
        return outputs


class QwenModel(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([QwenBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
  
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
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
        batch_size, seq_length = input_ids.size()
        hidden_states = self.wte(input_ids)  # Qwen无单独位置嵌入，依赖RoPE

        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            slot_mapping = slot_mappings[i] if slot_mappings else None
            block_table = block_tables[i] if block_tables else None

            outputs = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slot_mapping,
                block_table=block_table,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents += (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        return (hidden_states, presents) if use_cache else (hidden_states,)


class QwenLMHeadModel(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.transformer = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 共享词嵌入权重（Qwen通常如此）
        self.lm_head.weight = self.transformer.wte.weight

    def load_huggingface_weights(self, model_name: str):
        """加载Hugging Face预训练权重"""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()

        # 映射权重（处理命名差异）
        state_dict = {}
        for k, v in hf_state_dict.items():
            # Qwen的transformer对应到本模型的transformer
            if k.startswith("model."):
                state_dict[k.replace("model.", "transformer.")] = v
            # 映射lm_head
            elif k == "lm_head.weight":
                state_dict[k] = v

        self.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
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
        transformer_outputs = self.transformer(
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
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)
        outputs = (logits,) + transformer_outputs[1:]
        return outputs