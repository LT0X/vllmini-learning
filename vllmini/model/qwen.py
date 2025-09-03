import torch
from torch import nn
from transformers import AutoConfig
from typing import List, Optional, Tuple
from paged_attention_cuda import paged_attention_v1, cache_ops

class QwenAttention(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim **-0.5
        self.kv_channels = config.kv_channels
        self.num_key_value_heads = self.hidden_size // self.kv_channels  # 从配置推导KV头数

        # Qwen使用合并的c_attn（包含QKV），与GPT2类似但需适配GQA
        self.c_attn = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
            bias=not config.no_bias
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)

        # 旋转位置编码参数
        self.rotary_emb_base = config.rotary_emb_base
        self.rotary_pct = config.rotary_pct
        self.rotary_dim = int(self.head_dim * self.rotary_pct)

        self.block_size = 16  # 保持与原项目一致
        self.max_blocks = 1024

    def _rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """实现Qwen的旋转位置编码（基于配置参数）"""
        seq_len = q.shape[2]
        device = q.device

        # 生成频率
        inv_freq = 1.0 / (self.rotary_emb_base** (torch.arange(0, self.rotary_dim, 2, device=device) / self.rotary_dim))
        t = position_ids.unsqueeze(-1).float()  # [batch, seq_len, 1]
        freqs = torch.matmul(t, inv_freq.unsqueeze(0))  # [batch, seq_len, rotary_dim/2]
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # [batch, seq_len, rotary_dim]

        # 应用到Q和K
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]

        # 旋转操作
        q_rot = (q_rot * emb.cos()) + (self._rotate_half(q_rot) * emb.sin())
        k_rot = (k_rot * emb.cos()) + (self._rotate_half(k_rot) * emb.sin())

        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        return q, k

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

        # 计算QKV（Qwen的c_attn合并了QKV权重）
        qkv = self.c_attn(hidden_states)
        q_len = self.num_attention_heads * self.head_dim
        k_len = self.num_key_value_heads * self.head_dim
        q, k, v = qkv.split([q_len, k_len, k_len], dim=-1)

        # 重塑QKV
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        q, k = self._rotary_pos_emb(q, k, position_ids)

        # 缓存KV（保持与原项目缓存逻辑一致）
        if use_cache:
            self._cache_kv(k, v, key_cache, value_cache, slot_mapping)

        if is_prefill:
            # 预填充阶段使用普通注意力
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        else:
            # 解码阶段使用分页注意力
            attn_output = self._paged_attention(q, key_cache, value_cache, block_table, seq_lens, max_seq_len)
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.c_proj(attn_output)

        if use_cache:
            return attn_output, (k, v)
        return attn_output, None

    def _vanilla_attention(self, q, k, v, attention_mask):
        # 处理GQA的头映射（复制KV头以匹配Q头数量）
        if self.num_attention_heads != self.num_key_value_heads:
            k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    def _cache_kv(self, k: torch.Tensor, v: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor):
        # 适配原项目的缓存逻辑
        k = k.transpose(1, 2).contiguous().view(-1, self.num_key_value_heads, self.head_dim)
        v = v.transpose(1, 2).contiguous().view(-1, self.num_key_value_heads, self.head_dim)
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
        num_seqs, num_heads, seq_len, head_dim = q.shape
        q = q.view(-1, num_heads, head_dim)  # 适配分页注意力输入格式
        out = torch.empty_like(q)
        
        paged_attention_v1(
            out,
            q,
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
        return out

class QwenMLP(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # 匹配Qwen的MLP结构（w1, w2, c_proj）
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=not config.no_bias)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=not config.no_bias)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)  # 注意权重键中的c_proj
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.w2(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class QwenBlock(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = QwenAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QwenMLP(config)

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
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
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
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        outputs = hidden_states
        if use_cache:
            outputs = hidden_states, attn_outputs[1]

        return outputs

class QwenModel(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)  # 词嵌入
        # Qwen不使用单独的位置嵌入（依赖旋转编码），移除wpe
        self.h = nn.ModuleList([QwenBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,  # 用于旋转编码的位置ID
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
        hidden_states = self.wte(input_ids)  # 仅词嵌入，无位置嵌入
        presents = () if use_cache else None

        for i, block in enumerate(self.h):
            slot_mapping = slot_mappings[i] if slot_mappings is not None else None
            block_table = block_tables[i] if block_tables is not None else None

            outputs = block(
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
                position_ids=position_ids,  # 传递位置ID用于旋转编码
            )

            hidden_states = outputs[0]
            if use_cache:
                presents += (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

class QwenLMHeadModel(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config
        self.transformer = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 词表权重共享
        self.lm_head.weight = self.transformer.wte.weight

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

        outputs = (logits,)
        if use_cache:
            outputs += (transformer_outputs[1],)
        return outputs

    def load_huggingface_weights(self, model_path: str):
        """加载HuggingFace格式的Qwen权重（适配量化权重键）"""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 词嵌入权重
        self.transformer.wte.load_state_dict(hf_model.transformer.wte.state_dict())
        self.lm_head.load_state_dict(hf_model.lm_head.state_dict())
        
        # 层归一化与最终归一化
        self.transformer.ln_f.load_state_dict(hf_model.transformer.ln_f.state_dict())
        
        # 逐层加载权重
        for i in range(len(self.transformer.h)):
            # 注意力层
            self.transformer.h[i].ln_1.load_state_dict(hf_model.transformer.h[i].ln_1.state_dict())
            self.transformer.h[i].attn.c_attn.load_state_dict(hf_model.transformer.h[i].attn.c_attn.state_dict())
            self.transformer.h[i].attn.c_proj.load_state_dict(hf_model.transformer.h[i].attn.c_proj.state_dict())
            
            # MLP层
            self.transformer.h[i].ln_2.load_state_dict(hf_model.transformer.h[i].ln_2.state_dict())
            self.transformer.h[i].mlp.w1.load_state_dict(hf_model.transformer.h[i].mlp.w1.state_dict())
            self.transformer.h[i].mlp.w2.load_state_dict(hf_model.transformer.h[i].mlp.w2.state_dict())
            self.transformer.h[i].mlp.c_proj.load_state_dict(hf_model.transformer.h[i].mlp.c_proj.state_dict())
        
        print(f"Successfully loaded weights from {model_path}")