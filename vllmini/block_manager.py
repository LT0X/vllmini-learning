from typing import Dict, Tuple, List
import torch
from .kv_cache import KVCache  # 导入KV缓存管理类


class BlockManager:
    """
    块管理器：负责KV缓存块的分配、释放、换入换出（GPU与CPU之间），
    支持多序列并行生成时的缓存高效管理，是分页注意力（Paged Attention）的核心组件。
    """
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_size: int, max_blocks_per_seq:int):
        """
        初始化块管理器
        参数：
            num_blocks: 总缓存块数量（全局可用的块数）
            block_size: 每个块可存储的token数量（块大小）
            num_heads: 注意力头数量（与模型一致）
            head_size: 每个注意力头的维度（与模型一致）
            max_blocks_per_seq: 单个序列最多可占用的块数（限制单序列内存使用）
        """
        self.num_blocks = num_blocks  # 总块数
        self.block_size = block_size  # 每个块的token容量
        self.num_heads = num_heads    # 注意力头数
        self.head_size = head_size    # 每个头的维度

        self.max_blocks_per_seq = max_blocks_per_seq  # 新增这一行

        # 初始化GPU上的KV缓存（核心缓存，用于实时计算）
        self.kv_cache = KVCache(
            num_blocks=num_blocks,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq
        )
        # CPU缓存（用于临时存储换出的KV数据，缓解GPU内存压力）
        self.cpu_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[int, List[int], List[torch.Tensor], List[List[int]]]:
        """
        为预填充阶段（prefill）分配KV缓存块
        
        预填充阶段是对输入序列的首次处理，需要为整个序列分配初始缓存块。
        
        参数：
            seq_id: 序列ID（唯一标识一个生成任务）
            num_layers: 模型的Transformer层数
            seq_len: 输入序列的长度（token数量）
        
        返回：
            序列ID、分配的块列表、槽映射（token到缓存位置的映射）、分页注意力块表（块的索引信息）
        """
        # 调用KV缓存的预填充分配方法
        allocated, slot_mappings, paged_attention_block_table = self.kv_cache.allocate_for_prefill(seq_id, num_layers, seq_len)
        return seq_id, allocated, slot_mappings, paged_attention_block_table

    def get_block_table(self, seq_id: int) -> List[Tuple[int, int]]:
        """
        获取序列的块表（记录每个块的使用情况）
        
        块表是每个序列占用的缓存块的元数据，包含（块ID，已填充token数）的列表。
        
        参数：
            seq_id: 
            .序列ID
        
        返回：
            块表列表，每个元素为（块ID，该块已填充的token数）
        """
        return self.kv_cache.get_block_table(seq_id)

    def get_paged_attention_block_table(self, seq_id: int) -> List[List[int]]:
        """
        获取分页注意力块表（用于分页注意力计算的块索引映射）
        
        分页注意力块表记录了每个Transformer层中，块在缓存中的索引位置，
        是实现分页注意力（高效处理非连续缓存块）的关键数据结构。
        
        参数：
            seq_id: 序列ID
        
        返回：
            分页注意力块表（按层组织的块索引列表）
        """
        return self.kv_cache.get_paged_attention_block_table(seq_id)

    def get_kv_cache(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取序列当前的KV缓存张量（key缓存和value缓存）
        
        参数：
            seq_id: 序列ID
        
        返回：
            （key缓存张量，value缓存张量）
        """
        return self.kv_cache.get_kv_cache(seq_id)

    def decode_step(self, seq_id: int, input_len: int) -> Tuple[List[List[int]], torch.Tensor]:
        """
        解码步骤（生成下一个token时）的缓存管理
        
        解码阶段是逐token生成的过程，需要为新生成的token分配缓存空间，
        若当前块已满则分配新块，并更新块表和映射关系。
        
        参数：
            seq_id: 序列ID
            input_len: 本次输入的token长度（通常为1，因为逐token生成）
        
        返回：
            更新后的分页注意力块表、新的槽映射（新token的缓存位置）
        """
        # 获取当前序列的块表和分页注意力块表
        block_table = self.kv_cache.get_block_table(seq_id)
        paged_attention_block_table = self.kv_cache.get_paged_attention_block_table(seq_id)

        new_slot_mapping = []  # 新token的槽映射（缓存位置）
        # 遍历每个Transformer层的块表
        for layer_idx, layer_blocks in enumerate(paged_attention_block_table):
            last_block = -1  # 记录当前层的最后一个有效块
            # 查找最后一个填充的块（通过块表中的-1标记未使用位置）
            for i in range(1, len(layer_blocks[0])):
                if layer_blocks[0][i] == -1:
                    last_block = layer_blocks[0][i-1]  # 最后一个有效块ID
                    break
            
            # 查找最后一个块的详细信息（块ID和已填充数量）
            last_block_info = None
            for (block, filled) in block_table:
                if block == last_block:
                    last_block_info = (block, filled)
                    break
            
            _, num_filled = last_block_info  # 最后一个块已填充的token数

            # 若当前块已满（已填充数量达到块大小），则分配新块
            if num_filled == self.block_size:
                print("在BlockManager.decode_step中，块已填满，需要分配新块。")
                # 为当前层分配新块并添加到序列的块表中
                new_block = self.kv_cache.append_block(seq_id, layer_idx)
                last_block = new_block  # 更新最后一个块为新分配的块
                num_filled = 0  # 新块初始填充数量为0

            # 计算新token在缓存中的位置（块ID * 块大小 + 填充位置）
            new_slot = last_block * self.block_size + num_filled
            new_slot_mapping.append(torch.tensor([new_slot], dtype=torch.long, device="cuda"))
            
            # 更新块表中最后一个块的填充数量（+输入长度，通常为1）
            self.kv_cache.update_block_table(seq_id, last_block, num_filled + input_len)

        # 获取更新后的分页注意力块表/
        paged_attention_block_table = self.kv_cache.get_paged_attention_block_table(seq_id)        

        return paged_attention_block_table, new_slot_mapping

    def free(self, seq_id: int):
        """
        释放序列占用的所有缓存资源（GPU和CPU）
        
        当序列生成完成或被终止时，释放其占用的KV缓存块，避免资源泄漏。
        
        参数：
            seq_id: 序列ID
        """
        self.kv_cache.free(seq_id)  # 释放GPU上的KV缓存
        # 释放CPU缓存（如果存在）
        if seq_id in self.cpu_cache:
            del self.cpu_cache[seq_id]

    def swap_to_cpu(self, seq_id: int):
        """
        将序列的KV缓存从GPU换出到CPU（缓解GPU内存压力）
        
        当GPU内存不足时，可将暂时不处理的序列缓存换至CPU，释放GPU资源。
        
        参数：
            seq_id: 序列ID
        """
        # 获取GPU上的KV缓存张量
        key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)
        # 复制到CPU并存储在cpu_cache中
        self.cpu_cache[seq_id] = (key_cache.cpu(), value_cache.cpu())
        # 释放GPU上的缓存块
        self.kv_cache.free(seq_id)

    def swap_from_cpu(self, seq_id: int) -> bool:
        """
        将序列的KV缓存从CPU换回GPU（恢复处理）
        
        当需要继续处理之前换出到CPU的序列时，将其缓存换回GPU。
        
        参数：
            seq_id: 序列ID
        
        返回：
            换回成功则返回True，失败（如GPU内存不足）则返回False
        """
        # 若序列不在CPU缓存中，返回失败
        if seq_id not in self.cpu_cache:
            return False

        # 从CPU缓存中获取KV数据
        cpu_key_cache, cpu_value_cache = self.cpu_cache[seq_id]
        try:
            # 为序列重新在GPU上分配缓存块
            allocated = self.kv_cache.allocate(seq_id, cpu_key_cache.size(0))
            # 获取新分配的GPU缓存张量
            key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)
            # 将CPU数据复制到GPU缓存中
            key_cache.copy_(cpu_key_cache.cuda())
            value_cache.copy_(cpu_value_cache.cuda())
            # 移除CPU缓存中的数据（已换回GPU）
            del self.cpu_cache[seq_id]
            return True
        except RuntimeError: 
            # 若分配失败（如GPU内存不足），返回False
            return False
        
    def reback_kvcache(self, seq_id: int,back_length: int):
        """
        主模型拒绝了草稿，需要将对应的KV缓存进行回退
        """
        block_table =self.kv_cache.get_block_table(seq_id)
        paged_attention_block_table =self.kv_cache.get_paged_attention_block_table(seq_id)
        
        for i in range(back_length):
             # 遍历每个Transformer层的块表
            for layer_idx, layer_blocks in enumerate(paged_attention_block_table):
                last_block = -1  # 记录当前层的最后一个有效块
                paged_attention_block_index = None
                # 查找最后一个填充的块（通过块表中的-1标记未使用位置）
                for i in range(1, len(layer_blocks[0])):
                    if layer_blocks[0][i] == -1:
                        last_block = layer_blocks[0][i-1]  # 最后一个有效块ID
                        paged_attention_block_index = i - 1
                        break
                
                # 查找最后一个块的详细信息（块ID和已填充数量）
                last_block_info = None
                block_table_index = -1
                for (block, filled) in block_table:
                    if block == last_block:
                        last_block_info = (block, filled)
                        block_table_index +=1
                        break
                
                block_num, num_filled = last_block_info 
                if num_filled != 1:
                    #无需回收block,需要更新对应的元数据
                    self.kv_cache.update_block_table(seq_id, block_num, num_filled-1)
                else:
                    #token回退后，这个block填充0个token,需要进行回收
                    self.kv_cache.freeOneBlock(
                        seq_id=seq_id,
                        block_table_index= block_table_index,
                        block_num=block_num
                        
                    )
                    self.kv_cache.free_paged_attention_block_tables(seq_id,layer_idx,paged_attention_block_index)
                     
                    
                    
                    

                    

    



            

        
        

