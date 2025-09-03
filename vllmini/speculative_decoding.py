import torch
from typing import List, Tuple, Optional, Dict
from transformers import PreTrainedTokenizer
from vllmini.block_manager import BlockManager
from vllmini.model.helpers.generate_triangular_mask import generate_triangular_mask
import torch.nn.functional as F
from vllmini.model.qwen2 import Qwen2LMHeadModel
import copy


class LLMSpeculativeDecoding:
    def __init__(
        self,
        main_model: Qwen2LMHeadModel,
        draft_model:   Qwen2LMHeadModel,
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

    def  generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        is_last: bool,
        seq_id:int,
        sequence_lengths : int,
        
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """使用草稿模型生成候选token"""

         # 生成位置ID
        position_ids = torch.tensor([sequence_lengths], device=input_ids.device)
        
        paged_attention_block_table, new_slot_mappings = self.draft_block_manager.decode_step(seq_id, 1)
        key_cache, value_cache = (self.draft_block_manager.kv_cache.key_cache, 
                                 self.draft_block_manager.kv_cache.value_cache)
        
        
        
        logits, _ = self.draft_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=True,
            is_prefill=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=new_slot_mappings,
            block_tables=paged_attention_block_table,
            seq_lens=torch.tensor([sequence_lengths], dtype=torch.int32, device=input_ids.device),
            max_seq_len=self.draft_block_manager.kv_cache.max_blocks_per_seq * self.draft_block_manager.block_size
                    )
        if is_last:
            #开始清除draft的kv_cache缓存
            self.draft_block_manager.free(seq_id)
            print()     
        return logits
    def validate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: list,
        seq_id: int,
        draft_logits: list,
        sequence_length :int,
        first_main_logit: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """使用主模型验证候选token"""
        # 将输入和候选token拼接
        # all_tokens = torch.cat([input_ids, draft_tokens], dim=1)
        draft_len = len(draft_tokens)
        for i in range(draft_len):
            draft_tokens[i] = torch.clone(draft_tokens[i]).cpu()
        
        # 生成位置ID
        position_ids = torch.tensor([sequence_length], device=input_ids.device)
       
        #生成掩码
        main_attention_mask = generate_triangular_mask(1, self.main_block_manager.num_heads,draft_len)
        
        #分配需要的KVcache

        paged_attention_block_table =None , 
        new_slot_mapping = None
        for i in range(draft_len):
            paged_attention_block_table,new_slot_mapping = self.main_block_manager.decode_step(seq_id, 1)
        
        main_logits, _ = self.main_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=main_attention_mask,
                use_cache=True,
                is_prefill=True,
                key_cache=self.main_block_manager.kv_cache.key_cache,
                value_cache=self.main_block_manager.kv_cache.value_cache,
                slot_mappings=new_slot_mapping,
                block_tables=paged_attention_block_table,
                seq_lens=torch.tensor([sequence_length], dtype=torch.int32, device=input_ids.device),
                max_seq_len=self.main_block_manager.kv_cache.max_blocks_per_seq * self.main_block_manager.block_size
                    )
        
        last_logits = main_logits[:, -1, :]
        
        print(str(main_logits.shape)+"主模型logit 形状")
        
        for i in range(len(draft_tokens)):
            print("这个 tokens 是"+ str(draft_tokens[i]))
        
        accepted_length = 0
        temperature = 1.0  # 与采样函数保持一致的温度参数
        
        draft_logits = torch.cat(draft_logits,dim=1)   
        
        
        
        print("first_main_logit"+ str(first_main_logit.shape))
        main_logits = torch.cat([first_main_logit, main_logits[:, -draft_len-1:-1, :]], dim=1)
        # 应用温度缩放
        print(str(main_logits.shape)+"主模型logit 形状")
        print(str(draft_logits.shape)+"草稿logit 形状")

        main_logits_scaled = main_logits / temperature
        draft_logits_scaled = draft_logits / temperature
        
        # 计算概率分布
        # top_k = 50
        # top_k_logits, top_k_indices = torch.topk(logits, top_k)
        # main_probs = F.softmax(main_logits_scaled, dim=-1)
        # draft_probs = F.softmax(draft_logits_scaled, dim=-1)
        
        # 遍历每个草稿token，判断是否接受
        for i in range(draft_len):
            
           
            main_probs = F.softmax(main_logits_scaled[:,i,:], dim=-1)
            

            draft_probs = F.softmax(draft_logits_scaled[:,i,:], dim=-1)
            
            current_token = draft_tokens[i]
            print("当前token:",str(current_token))
            # 1. 获取主模型和草稿模型对当前token的概率
            main_prob = main_probs[0, current_token]
            draft_prob = draft_probs[0,current_token]
            print("shape main"+str(main_probs.shape))
            print(str(main_probs.shape)+"pro主模型")
            print(str(draft_probs.shape)+"pro草稿模型") 
            print("主模型概率:",main_prob)
            print("草稿模型概率:",draft_prob)

            
            # torch.save(main_probs, "/home/xtc/project/vllmini/vllmini/doc/main_probs.txt")
            # torch.save(draft_probs, "/home/xtc/project/vllmini/vllmini/doc/draft_probs.txt")


            # 2. 计算接受概率：基于主模型与草稿模型的概率比，确保不超过1.0
            # 这种方式更严谨，考虑了两个模型的概率分布对比
            accept_prob = torch.min(torch.tensor(1.0, device=self.device), main_prob / draft_prob)
            
            # 3. 额外检查：如果主模型对该token的概率过低，直接拒绝
            # 避免接受主模型本身认为极不可能的token
            min_accept_prob = 0.01  # 可调整的最小接受概率阈值
            if main_prob < min_accept_prob:
               accepted_length = 0
               break
            
            # 4. 执行随机采样决定是否接受
            if torch.rand(1, device=self.device) < accept_prob:
                accepted_length += 1
            else:
                break
            
        
        # 如果没有接受任何token，使用主模型生成一个token
        if accepted_length == 0:
            # 复用已有的采样函数，保持一致性
            return 0,None

        #开始处理KV缓存
        if accepted_length !=draft_len:
            self.main_block_manager.reback_kvcache(seq_id,draft_len)
        
        return accepted_length, last_logits
       
    

    def prefill(self,seq_id:int,
                seq_len:int,
                input_ids:torch.Tensor,
                model_type:int):
        

        if model_type == 0:
            #分配主模型prefillKV缓存块
            num_layers = len(self.main_model.model.layers)
            seq_id,_, main_slot_mappings,page_attention_block_tables = self.main_block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
            
            #获取分配的缓存
            main_key_cache, main_value_cache = (self.main_block_manager.kv_cache.key_cache, 
                                               self.main_block_manager.kv_cache.value_cache)
            
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
                block_tables=page_attention_block_tables,
            )
            return main_logits
        elif model_type == 1:
            #分配prefillKV缓存块
            num_layers = len(self.draft_model.model.layers)
            seq_id,_, draft_slot_mappings, page_attention_block_tables = self.draft_block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
            
            #获取分配的缓存
            draft_key_cache, draft_value_cache = (self.draft_block_manager.kv_cache.key_cache,
                                                self.draft_block_manager.kv_cache.value_cache)
            
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
                block_tables=page_attention_block_tables,
            )
            return draft_logits
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
    
    

        

        




        
