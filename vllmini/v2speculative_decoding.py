# vllmini/speculative_decoding.py
import torch
from typing import List, Tuple, Optional, Dict

class LLMSpeculativeDecoding1:
    def __init__(self, main_model, draft_model, tokenizer, max_speculative_steps: int = 5):
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.max_speculative_steps = max_speculative_steps  # 草稿模型一次生成的最大token数
        self.eos_token_id = tokenizer.eos_token_id

    def generate_candidates(self, input_ids: torch.Tensor, seq_len: int, max_length: int) -> Tuple[torch.Tensor, int]:
        """使用草稿模型生成候选token序列"""
        candidates = input_ids.clone()
        current_len = seq_len
        
        # 生成最多max_speculative_steps个候选token
        for _ in range(self.max_speculative_steps):
            if current_len >= max_length:
                break
                
            with torch.no_grad():
                outputs = self.draft_model(candidates)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                candidates = torch.cat([candidates, next_token], dim=-1)
                current_len += 1
                
                if next_token.item() == self.eos_token_id:
                    break
                    
        # 返回候选序列和生成的token数量
        return candidates, current_len - seq_len

    def verify_candidates(self, input_ids: torch.Tensor, candidates: torch.Tensor, 
                         seq_len: int, candidate_len: int, 
                         key_cache: torch.Tensor, value_cache: torch.Tensor,
                         slot_mappings: List[torch.Tensor], block_tables: List[torch.Tensor]) -> Tuple[int, torch.Tensor]:
        """使用主模型验证候选token"""
        if candidate_len == 0:
            return 0, input_ids
            
        # 主模型一次性验证所有候选token
        with torch.no_grad():
            outputs = self.main_model(
                input_ids=candidates,
                use_cache=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mappings,
                block_tables=block_tables
            )
            
            logits = outputs.logits
            # 计算每个候选token的接受概率
            probs = torch.softmax(logits, dim=-1)
            
            # 找到第一个不被接受的token位置
            accepted_length = 0
            for i in range(seq_len, seq_len + candidate_len):
                candidate_token = candidates[0, i].item()
                prob = probs[0, i-1, candidate_token].item()
                
                # 简单的接受准则：概率大于随机阈值
                if prob > 1.0 / self.main_model.config.vocab_size:
                    accepted_length += 1
                else:
                    break
                    
            # 如果所有候选都被接受，主模型再额外生成一个token
            if accepted_length == candidate_len:
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                accepted_length += 1
                final_ids = torch.cat([candidates, next_token], dim=-1)
            else:
                # 只接受到第一个不被接受的token之前的序列
                final_ids = candidates[:, :seq_len + accepted_length]
                
            return accepted_length, final_ids