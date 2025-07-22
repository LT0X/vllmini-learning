import torch
import time
from typing import Dict, List, Tuple


class KVCache:
    """
    KV缓存管理器：实现基于块（block）的键值（Key-Value）缓存管理，
    支持预填充（prefill）和逐token解码（decode）阶段的缓存分配与更新，
    是分页注意力（Paged Attention）机制的核心组件，用于高效管理GPU内存。
    """
    def __init__(self, num_blocks: int, num_heads: int, head_size: int, block_size: int, max_blocks_per_seq:int):
        """
        初始化KV缓存
        
        参数：
            num_blocks: 总缓存块数量（全局可用的块数）
            num_heads: 注意力头数量（与模型一致）
            head_size: 每个注意力头的维度（与模型一致）
            block_size: 每个块可存储的token数量（块大小）
            max_blocks_per_seq: 单个序列最多可占用的块数（限制单序列内存使用）
        """
        self.num_blocks = num_blocks  # 总块数
        self.num_heads = num_heads    # 注意力头数
        self.head_size = head_size    # 每个头的维度
        self.block_size = block_size  # 每个块的token容量
        self.max_blocks_per_seq = max_blocks_per_seq  # 单序列最大块数限制

        # 初始化Key缓存张量（特殊形状用于显存优化，实际等效于[num_blocks, num_heads, block_size, head_size]）
        # 形状：[块数, 注意力头数, head_size//8, 块大小, 8]（float16占2字节，此处通过拆分维度优化访问效率）
        self.key_cache = torch.zeros(
            num_blocks, num_heads, head_size // 8, block_size, 8, 
            dtype=torch.float16, device='cuda'
        )
        # 初始化Value缓存张量（形状：[块数, 注意力头数, 头维度, 块大小]）
        self.value_cache = torch.zeros(
            num_blocks, num_heads, head_size, block_size, 
            dtype=torch.float16, device='cuda'
        )

        # 空闲块列表（记录当前未被使用的块ID）
        self.free_blocks = list(range(num_blocks))
        # 已分配块映射（序列ID -> 该序列占用的块ID列表）
        self.allocated_blocks: Dict[int, List[int]] = {}
        # 块表（序列ID -> [(块ID, 已填充token数), ...]，记录每个块的使用状态）
        self.block_tables: Dict[int, List[Tuple[int, int]]] = {}
        # 分页注意力块表（序列ID -> 按层组织的块索引列表，用于分页注意力计算）
        # 结构：[ [layer0的块列表], [layer1的块列表], ... ]，未使用位置用-1填充
        self.paged_attention_block_tables: Dict[int, List[List[int]]] = {}

    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[List[int], List[torch.Tensor], List[List[int]]]:
        """
        为预填充阶段（prefill）分配KV缓存块
        预填充阶段是对输入序列的首次处理（如用户输入的prompt），需要为每个Transformer层分配初始块，
        存储该层的注意力键值对（KV）。
         
        参数：
            seq_id: 序列唯一标识ID
            num_layers: 模型的Transformer层数（每个层需要独立的KV缓存块）
            seq_len: 输入序列的长度（token数量）
        
        返回：
            分配的块ID列表、槽映射（token到缓存位置的映射）、分页注意力块表
        """
        # 检查是否有足够的空闲块（至少需要与层数相同的块数）
        if len(self.free_blocks) < num_layers:
            raise RuntimeError("预填充分配时没有足够的空闲块")

        # 从空闲块中分配num_layers个块（取前num_layers个）
        allocated = self.free_blocks[:num_layers]
        self.free_blocks = self.free_blocks[num_layers:]  # 更新空闲块列表
        self.allocated_blocks[seq_id] = allocated  # 记录序列分配的块
        
        # 初始化块表：每个块的已填充数量为min(序列长度, 块大小)（块可能装不下整个序列）
        self.block_tables[seq_id] = [(block, min(seq_len, self.block_size)) for block in allocated]
        # 初始化分页注意力块表：每个层对应一个块列表，用-1填充未使用的位置（最多max_blocks_per_seq个块）
        self.paged_attention_block_tables[seq_id] = [
            torch.tensor(
                [[block] + [-1] * (self.max_blocks_per_seq - 1)],  # 结构：[块ID, -1, -1, ...]
                device="cuda", dtype=torch.int32
            ) for block in allocated
        ]

        # 计算槽映射（slot mappings）：每个token在缓存中的位置（块ID * 块大小 + token在块内的偏移）
        slot_mappings = [
            torch.arange(seq_len, dtype=torch.long, device='cuda') + block * self.block_size 
            for block in allocated
        ]

        return allocated, slot_mappings, self.paged_attention_block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> List[Tuple[int, int]]:
        """获取序列的块表（记录每个块的ID和已填充token数）"""
        if seq_id not in self.block_tables:
            raise ValueError(f"序列 {seq_id} 不存在块表")
        return self.block_tables[seq_id]

    def get_paged_attention_block_table(self, seq_id: int) -> List[List[int]]:
        """获取序列的分页注意力块表（按层组织的块列表）"""
        if seq_id not in self.paged_attention_block_tables:
            raise ValueError(f"序列 {seq_id} 不存在分页注意力块表")
        return self.paged_attention_block_tables[seq_id]

    def get_kv_cache(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据序列ID获取对应的KV缓存张量（key和value）"""
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"序列 {seq_id} 未分配KV缓存")

        # 通过已分配的块ID索引到全局KV缓存中，获取该序列的缓存
        blocks = self.allocated_blocks[seq_id]
        return self.key_cache[blocks], self.value_cache[blocks]

    def append_block(self, seq_id: int, layer_idx: int) -> int:
        """
        在解码阶段为序列分配新的块（当现有块已满时）
        
        参数：
            seq_id: 序列ID
            layer_idx: 需要分配新块的Transformer层索引
        
        返回：
            新分配的块ID
        """
        # 检查是否有空闲块
        if len(self.free_blocks) == 0:
            raise RuntimeError("没有可用的空闲块用于扩展")
        
        # 从空闲块中取一个新块
        new_block = self.free_blocks.pop(0)
        self.allocated_blocks[seq_id].append(new_block)  # 更新序列的已分配块列表
        
        # 更新块表：新增（新块ID, 0）（初始已填充数量为0）
        self.block_tables[seq_id].append((new_block, 0))
        # 找到分页注意力块表中该层的第一个未使用位置（-1），填入新块ID
        new_block_index = 0
        for i in range(self.max_blocks_per_seq):
            if self.paged_attention_block_tables[seq_id][layer_idx][0][i] == -1:
                new_block_index = i
                break
        self.paged_attention_block_tables[seq_id][layer_idx][0][new_block_index] = new_block

        return new_block

    def update_block_table(self, seq_id: int, block: int, num_filled: int):
        """
        更新块表中指定块的已填充数量
        
        参数：
            seq_id: 序列ID
            block: 块ID
            num_filled: 新的已填充token数量
        """
        # 遍历块表，找到目标块并更新已填充数量
        for i, (b, _) in enumerate(self.block_tables[seq_id]):
            if b == block:
                self.block_tables[seq_id][i] = (block, num_filled)
                break

    def free(self, seq_id: int):
        """
        释放序列占用的所有KV缓存块
        
        参数：
            seq_id: 序列ID
        """
        if seq_id in self.allocated_blocks:
            # 将序列占用的块归还给空闲块列表
            self.free_blocks.extend(self.allocated_blocks[seq_id])
            # 删除序列的所有记录
            del self.allocated_blocks[seq_id]
            del self.block_tables[seq_id]
            del self.paged_attention_block_tables[seq_id]