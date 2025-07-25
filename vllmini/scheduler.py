import time
from queue import PriorityQueue
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from .block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from vllmini.model.helpers.generate_triangular_mask import generate_triangular_mask

class Scheduler:

    
    def __init__(self, model:GPT2LMHeadModel, block_manager:BlockManager, max_length: int):
        self.model = model
        self.model = self.model.to(torch.float16)  # 使用半精度浮点数减少内存占用
        self.block_manager = block_manager  # 负责KV缓存块的分配和管理
        self.max_length = max_length  # 序列的最大生成长度
        self.queue = PriorityQueue()  # 待处理序列的优先级队列
        self.active_sequences: Dict[int, float] = {}  # 活跃序列及其到达时间
        self.last_logits: Dict[int, torch.Tensor] = {}  # 每个序列的最后一个时间步的logits
        self.sequence_lengths: Dict[int, int] = {}  # 每个序列的当前长度
        self.sequences: Dict[int, torch.Tensor] = {}  # 每个序列的完整token ID

    def add_sequence(self, input_ids: torch.Tensor):
        """添加新序列到调度器"""
        arrival_time = time.time()
        seq_id = self._generate_seq_id()
        self.queue.put((arrival_time, seq_id))
        self.active_sequences[seq_id] = arrival_time
        
        seq_len = input_ids.size(1)
        num_layers = len(self.model.transformer.h)
        
        # 为prefill阶段分配KV缓存块
        seq_id, _, slot_mappings, paged_attention_block_table = self.block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
        key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache
        
        # 生成因果掩码
        attention_mask = generate_triangular_mask(1, self.block_manager.num_heads, seq_len)

        # 执行prefill（首次前向传播）
        logits, _ = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(seq_len, device=input_ids.device),
            attention_mask=attention_mask,
            use_cache=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=paged_attention_block_table,
        )
        
        self.last_logits[seq_id] = logits[:, -1, :]
        self.sequence_lengths[seq_id] = seq_len
        self.sequences[seq_id] = input_ids
        print(f"P  refill complete for sequence {seq_id}, length: {seq_len}")
        return seq_id

    def run(self):
        """运行调度循环，处理队列中的所有序列"""
        while not self.queue.empty():
            print(f"Active sequences: {self.active_sequences}")
            print(f"Queue size: {self.queue.qsize()}")
            
            _, seq_id = self.queue.get()
            print(f"Processing sequence {seq_id}")

            if seq_id not in self.active_sequences:
                print(f"Sequence {seq_id} is no longer active, skipping")
                continue

            try:
                print("current sequence_lengths in run:", self.sequence_lengths)
                print(f"Processing sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                
                # 检查是否达到最大长度
                if self.sequence_lengths[seq_id] >= self.max_length:
                    print(f"Sequence {seq_id} has reached max_length, ending generation")
                    self.remove_sequence_from_processing(seq_id)
                    continue

                # 采样下一个token
                next_token = self.sample_next_token(seq_id)
                current_sequence = self.sequences[seq_id]
                input_ids = next_token.unsqueeze(0)
                self.sequences[seq_id] = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=-1)

                # 生成位置ID
                position_ids = torch.tensor([self.sequence_lengths[seq_id]], device=input_ids.device)
                
                # 为decode步骤分配新的KV缓存块
                paged_attention_block_table, new_slot_mappings = self.block_manager.decode_step(seq_id, 1)
                key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache

                # 执行decode（后续前向传播，使用KV缓存）
                logits, _ = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                    use_cache=True,
                    is_prefill=False,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mappings=new_slot_mappings,
                    block_tables=paged_attention_block_table,
                    seq_lens=torch.tensor([self.sequence_lengths[seq_id]], dtype=torch.int32, device=input_ids.device),
                    max_seq_len=self.block_manager.kv_cache.max_blocks_per_seq * self.block_manager.block_size
                )

                self.last_logits[seq_id] = logits[:, -1, :]
                self.sequence_lengths[seq_id] += 1

                # 检查是否完成生成（遇到EOS或达到最大长度）
                if next_token.item() != self.model.config.eos_token_id and self.sequence_lengths[seq_id] < self.max_length:
                    self.queue.put((self.active_sequences[seq_id], seq_id))
                    print(f"Re-queued sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                else:
                    print(f"Sequence {seq_id} completed or reached max_length, final length: {self.sequence_lengths[seq_id]}")
                    self.remove_sequence_from_processing(seq_id)

            except RuntimeError as e:
                print(f"Error processing sequence {seq_id}: {str(e)}")
                if "CUDA out of memory" in str(e):
                    self.handle_out_of_memory([seq_id])
                else:
                    raise e

    def handle_out_of_memory(self, batch_seq_ids: List[int]):
        """处理内存不足的情况，这里按理说应该是移除最老的序列以释放内存，
        但代码的实现是移除了最新的时间戳序列id,那移除的是最新的序列任务了
        """
        print("Handling out of memory")
        if self.active_sequences:
            
            seq_to_remove = max(
                (seq for seq in self.active_sequences if seq not in batch_seq_ids),
                key=self.active_sequences.get,
                default=None
            )
            if seq_to_remove is None:
                seq_to_remove = max(self.active_sequences, key=self.active_sequences.get)
            print(f"Removing sequence {seq_to_remove} due to memory constraints")
            self.remove_sequence_from_processing(seq_to_remove)
        else:
            print("No active sequences to remove")

    def remove_sequence_from_processing(self, seq_id: int):
        """从处理中移除序列，释放相关资源"""
        self.block_manager.free(seq_id)
        del self.active_sequences[seq_id]
        if seq_id in self.last_logits:
            del self.last_logits[seq_id]
        if seq_id in self.sequence_lengths:
            del self.sequence_lengths[seq_id]
    
    def remove_completed_sequence(self, seq_id:int):
        """移除已完成的序列，释放完整序列数据"""
        if seq_id in self.sequences:
            del self.sequences[seq_id]

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

    def _generate_seq_id(self) -> int:
        """生成唯一的序列ID"""
        
        return int(time.time() * 1000000)

