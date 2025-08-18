import torch
from typing import List, Tuple, Optional, Dict
from transformers import PreTrainedTokenizer
from vllmini.block_manager import BlockManager
from vllmini.model.helpers.generate_triangular_mask import generate_triangular_mask
import torch.nn.functional as F
class LLMSpeculativeDecoding:
    def __init__(
        self,
        main_model: torch.nn.Module,
        draft_model: torch.nn.Module,
        main_block_manager: BlockManager,
        draft_block_manager: BlockManager,
        tokenizer: PreTrainedTokenizer,
        max_speculative_steps: int = 5,
        eos_token_id: int = 50256  # GPT2的EOS token
        
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.max_speculative_steps = max_speculative_steps
        self.eos_token_id = eos_token_id
        self.main_block_manager = main_block_manager
        self.draft_block_manager = draft_block_manager
        
        # 确保模型在同一设备
        self.device = next(main_model.parameters()).device
        self.draft_model = self.draft_model.to(self.device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        block_manager,
        seq_id: int,  # 明确传入seq_id
        **kwargs
    ) -> torch.Tensor:
        """使用投机解码生成文本"""
        generated = input_ids.clone()
        current_length = input_ids.shape[1]
        
        # 为当前序列分配初始块
        num_layers = len(self.main_model.transformer.h)
        _, _, slot_mappings, block_tables = block_manager.allocate_for_prefill(
            seq_id, num_layers, current_length
        )
        
        while current_length < max_length:
            # 1. 草稿模型生成候选token
            draft_tokens, new_slot_mappings, new_block_tables = self._generate_draft_tokens(
                generated, 
                max_speculative_steps=min(self.max_speculative_steps, max_length - current_length),
                block_manager=block_manager,
                seq_id=seq_id,
                slot_mappings=slot_mappings,
                block_tables=block_tables,** kwargs
            )
            
            # 2. 主模型验证候选token
            accepted_length, new_tokens = self._validate_draft_tokens(
                generated, draft_tokens, 
                block_manager=block_manager,
                seq_id=seq_id,
                slot_mappings=new_slot_mappings,
                block_tables=new_block_tables,
                **kwargs
            )
            
            # 3. 更新生成结果
            generated = torch.cat([generated, new_tokens[:accepted_length]], dim=1)
            current_length += accepted_length
            
            # 更新缓存映射
            slot_mappings = new_slot_mappings
            block_tables = new_block_tables
            
          # 修改 speculative_decodingv3.py 第72行附近的代码
            if accepted_length > 0:
                # 确保索引不越界
                if accepted_length <= len(new_tokens):
                    if new_tokens[accepted_length-1] == self.eos_token_id:
                        # 处理EOS逻辑
                        break
                else:
                    # 日志提示或修正accepted_length（例如取最小值）
                    accepted_length = min(accepted_length, len(new_tokens))
                            
                    # 释放序列占用的块
                    block_manager.free(seq_id)
                    return generated
    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        max_speculative_steps: int,
        block_manager,
        seq_id: int,
        slot_mappings: List[torch.Tensor],
        block_tables: List[torch.Tensor],** kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """使用草稿模型生成候选token"""
        draft_tokens = []
        current_input = input_ids.clone()
        current_slot_mappings = slot_mappings
        current_block_tables = block_tables
        
        for _ in range(max_speculative_steps):
            batch_size, seq_len = current_input.shape
            position_ids = torch.tensor([seq_len - 1], device=self.device)
            
            # 草稿模型前向传播 (移除了seq_id参数)
            logits, _ = self.draft_model(
                input_ids=current_input[:, -1:],  # 只输入最后一个token
                position_ids=position_ids,
                use_cache=True,
                is_prefill=False,
                key_cache=block_manager.kv_cache.key_cache,
                value_cache=block_manager.kv_cache.value_cache,
                slot_mappings=current_slot_mappings,
                block_tables=current_block_tables,
                seq_lens=torch.tensor([seq_len - 1], dtype=torch.int32, device=self.device),
                max_seq_len=block_manager.max_blocks_per_seq * block_manager.block_size,
                **kwargs
            )
            
            # 采样下一个token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            draft_tokens.append(next_token)
            
            # 检查是否生成了EOS token
            if next_token.item() == self.eos_token_id:
                break
                
            # 更新输入和缓存映射
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            current_block_tables, current_slot_mappings = block_manager.decode_step(seq_id, 1)
            
        return (
            torch.cat(draft_tokens, dim=0).unsqueeze(0),
            current_slot_mappings,
            current_block_tables
        )
    def _generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        max_speculative_steps: int,
        block_manager,
        seq_id: int,
        slot_mappings: List[torch.Tensor],
        block_tables: List[torch.Tensor],** kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """使用草稿模型生成候选token"""
        draft_tokens = []
        current_input = input_ids.clone()
        current_slot_mappings = slot_mappings
        current_block_tables = block_tables
        
        for _ in range(max_speculative_steps):
            batch_size, seq_len = current_input.shape
            position_ids = torch.tensor([seq_len - 1], device=self.device)
            
            # 草稿模型前向传播 (移除了seq_id参数)
            logits, _ = self.draft_model(
                input_ids=current_input[:, -1:],  # 只输入最后一个token
                position_ids=position_ids,
                use_cache=True,
                is_prefill=False,
                key_cache=block_manager.kv_cache.key_cache,
                value_cache=block_manager.kv_cache.value_cache,
                slot_mappings=current_slot_mappings,
                block_tables=current_block_tables,
                seq_lens=torch.tensor([seq_len - 1], dtype=torch.int32, device=self.device),
                max_seq_len=block_manager.max_blocks_per_seq * block_manager.block_size,
                **kwargs
            )
            
            # 采样下一个token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            draft_tokens.append(next_token)
            
            # 检查是否生成了EOS token
            if next_token.item() == self.eos_token_id:
                break
                
            # 更新输入和缓存映射
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            current_block_tables, current_slot_mappings = block_manager.decode_step(seq_id, 1)
            
        return (
            torch.cat(draft_tokens, dim=0).unsqueeze(0),
            current_slot_mappings,
            current_block_tables
        )

    def _validate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        block_manager,
        seq_id: int,
        slot_mappings: List[torch.Tensor],
        block_tables: List[torch.Tensor],** kwargs
    ) -> Tuple[int, torch.Tensor]:
        """使用主模型验证候选token"""
        # 将输入和候选token拼接
        all_tokens = torch.cat([input_ids, draft_tokens], dim=1)
        input_len = input_ids.shape[1]
        draft_len = draft_tokens.shape[1]
        
        # 主模型前向传播 (移除了seq_id参数)
        logits, _ = self.main_model(
            input_ids=all_tokens[:, input_len:],  # 只处理新增的token
            position_ids=torch.arange(input_len, input_len + draft_len, device=self.device),
            use_cache=True,
            is_prefill=False,
            key_cache=block_manager.kv_cache.key_cache,
            value_cache=block_manager.kv_cache.value_cache,
            slot_mappings=slot_mappings,
            block_tables=block_tables,
            seq_lens=torch.tensor([input_len], dtype=torch.int32, device=self.device),
            max_seq_len=block_manager.max_blocks_per_seq * block_manager.block_size,
            **kwargs
        )
        
        # 计算接受长度
        probs = torch.softmax(logits, dim=-1)
        main_probs = probs[:, :, :]
        draft_token_ids = draft_tokens.squeeze(0)
        
        # 计算每个候选token的接受概率
        accepted = 0
        for i in range(draft_len):
            token_prob = main_probs[0, i, draft_token_ids[i]]
            if torch.rand(1, device=self.device) < token_prob:
                accepted += 1
            else:
                break
                
        # 如果所有候选都被接受，再采样一个token
        if accepted == draft_len:
            final_token = torch.argmax(probs[:, -1, :], dim=-1)
            draft_tokens = torch.cat([draft_tokens, final_token.unsqueeze(0)], dim=1)
            accepted += 1
            
        return accepted, draft_tokens
    

    def prefill(self,seq_id:int,num_layers:int, seq_len:int,input_ids:torch.Tensor,model_type:int):
        

        if model_type == 0:
            #分配prefillKV缓存块
            seq_id, main_slot_mappings, main_block_tables = self.main_block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
            
            #获取分配的缓存
            main_key_cache, main_value_cache = self.main_block_manager.kv_cache.key_cache, 
            self.main_block_manager.kv_cache.value_cache
            
            #生成掩码
            main_attention_mask = generate_triangular_mask(1, self.main_block_manager.num_heads, seq_len)

            #主模型进行前向传播
            main_logits,_ = self.main_model(
                input_ids=input_ids,
                position_ids=torch.arange(seq_len, device=input_ids.device),
                attention_mask=main_attention_mask,
                use_cache=True,
                key_cache=main_key_cache,
                value_cache=main_value_cache,
                slot_mappings=main_slot_mappings,
                block_tables=main_block_tables,
            )
            return main_logits
        elif model_type == 1:
            #分配prefillKV缓存块
            seq_id, draft_slot_mappings, draft_block_tables = self.draft_block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
            
            #获取分配的缓存
            draft_key_cache, draft_value_cache = self.draft_block_manager.kv_cache.key_cache,
            self.draft_block_manager.kv_cache.value_cache
            
            #生成掩码
            
            draft_attention_mask = generate_triangular_mask(1, self.draft_block_manager.num_heads, seq_len)
            draft_logits, _ = self.draft_model(
                input_ids=input_ids,
                position_ids=torch.arange(seq_len, device=input_ids.device),
                attention_mask=draft_attention_mask,
                use_cache=True,
                key_cache=draft_key_cache, 
                value_cache=draft_value_cache,
                slot_mappings=draft_slot_mappings,
                block_tables=draft_block_tables,
            )
            return draft_logits
    def sample_next_token(self, seq_id: int) -> torch.Tensor:
        """从模型输出的logits中采样下一个token"""
        logits = self.last_logits[seq_id]
        temperature = 1.0
        logits = logits / temperature
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[0, next_token_index[0]]
        return next_token
        

        




        
