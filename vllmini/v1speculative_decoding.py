import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

class LLMSpeculativeDecoding:
    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_speculative_steps: int = 5,
        eos_token_id: int = None
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.max_speculative_steps = max_speculative_steps  # 草稿模型最多生成的候选token数
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id
        
        # 确保模型在相同设备
        self.device = target_model.device
        self.draft_model = draft_model.to(self.device)

    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        seq_len: int,
        slot_mappings: List[torch.Tensor],
        block_tables: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """用草稿模型生成候选token序列"""
        candidates = input_ids.clone()
        current_len = seq_len
        draft_slot_mappings = slot_mappings
        draft_block_tables = block_tables

        for _ in range(self.max_speculative_steps):
            # 草稿模型单步生成
            position_ids = torch.tensor([current_len], device=self.device)
            logits, _ = self.draft_model(
                input_ids=candidates[:, -1:],
                position_ids=position_ids,
                attention_mask=None,
                use_cache=True,
                is_prefill=False,
                key_cache=self.draft_model.kv_cache.key_cache,
                value_cache=self.draft_model.kv_cache.value_cache,
                slot_mappings=draft_slot_mappings,
                block_tables=draft_block_tables,
                seq_lens=torch.tensor([current_len], dtype=torch.int32, device=self.device),
                max_seq_len=self.draft_model.kv_cache.max_blocks_per_seq * self.draft_model.block_size
            )
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            candidates = torch.cat([candidates, next_token.unsqueeze(0)], dim=-1)
            
            # 更新缓存映射
            draft_block_tables, draft_slot_mappings = self.draft_model.block_manager.decode_step(
                seq_id=id(candidates),  # 临时用张量地址作为seq_id
                input_len=1
            )
            
            current_len += 1
            if next_token.item() == self.eos_token_id:
                break

        return candidates, draft_slot_mappings, draft_block_tables

    def verify_candidates(
        self,
        input_ids: torch.Tensor,
        candidates: torch.Tensor,
        seq_len: int,
        target_slot_mappings: List[torch.Tensor],
        target_block_tables: List[torch.Tensor]
    ) -> Tuple[int, torch.Tensor]:
        """用主模型验证候选序列，返回接受的token数和最终logits"""
        num_candidates = candidates.size(1) - seq_len
        if num_candidates == 0:
            return 0, None

        # 主模型一次验证所有候选
        position_ids = torch.arange(seq_len, seq_len + num_candidates, device=self.device)
        logits, _ = self.target_model(
            input_ids=candidates,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=True,
            is_prefill=False,
            key_cache=self.target_model.kv_cache.key_cache,
            value_cache=self.target_model.kv_cache.value_cache,
            slot_mappings=target_slot_mappings,
            block_tables=target_block_tables,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            max_seq_len=self.target_model.kv_cache.max_blocks_per_seq * self.target_model.block_size
        )

        # 计算接受长度（找到模型与主模型预测一致的最大前缀）
        accept_length = 0
        for i in range(num_candidates):
            draft_token = candidates[0, seq_len + i]
            target_token = torch.argmax(logits[0, i], dim=-1)
            if draft_token == target_token:
                accept_length += 1
                if draft_token == self.eos_token_id:
                    break
            else:
                break

        return accept_length, logits[:, accept_length, :]