import unittest
import torch
from transformers import GPT2Tokenizer, GPT2Config
from vllmini.block_manager import BlockManager
from vllmini.speculative_decoding import LLMSpeculativeDecoding  # 替换为实际模块名
from vllmini.model.gpt2 import GPT2LMHeadModel

class TestLLMSpeculativeDecoding(unittest.TestCase):
    def setUp(self):
        # 初始化配置
        print("初始化")
        self.config = GPT2Config.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        
      
        self.main_model = GPT2LMHeadModel(self.config)
        self.draft_model = GPT2LMHeadModel(self.config)
        self.device = torch.device("cuda")  
        self.main_model = self.main_model.to(self.device)
        self.draft_model = self.draft_model.to(self.device)
        
        # 初始化块管理器（适配GPT2的参数）
        self.num_blocks = 100
        self.block_size = 16
        self.num_heads = self.config.num_attention_heads
        self.head_size = self.config.hidden_size // self.num_heads
        self.max_blocks_per_seq = 4
        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
            max_blocks_per_seq=self.max_blocks_per_seq
        )
        
        # 初始化投机解码组件
        self.speculative_decoder = LLMSpeculativeDecoding(
            main_model=self.main_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            max_speculative_steps=3,
            eos_token_id=self.eos_token_id
        )
        
        # 测试用输入
        self.prompt = "Hello world"
        self.input_ids = self.tokenizer.encode(
            self.prompt, 
            return_tensors="pt"
        ).to(self.device)
        self.seq_id = 123  # 测试用序列ID

    def test_generate_draft_tokens_basic(self):
        """测试草稿模型生成候选token的基本功能"""
        # 预填充块管理器
        num_layers = len(self.main_model.transformer.h)
        _, _, slot_mappings, block_tables = self.block_manager.allocate_for_prefill(
            self.seq_id, num_layers, self.input_ids.shape[1]
        )
        
        # 生成候选token
        draft_tokens, new_slot_mappings, new_block_tables = self.speculative_decoder._generate_draft_tokens(
            input_ids=self.input_ids,
            max_speculative_steps=3,
            block_manager=self.block_manager,
            seq_id=self.seq_id,
            slot_mappings=slot_mappings,
            block_tables=block_tables
        )
        
        # 验证候选token形状（[1, n]，n≤3）
        self.assertEqual(draft_tokens.ndim, 2)
        self.assertEqual(draft_tokens.shape[0], 1)
        self.assertLessEqual(draft_tokens.shape[1], 3)
        
        # 验证缓存映射不为空
        self.assertIsNotNone(new_slot_mappings)
        self.assertIsNotNone(new_block_tables)
        self.assertEqual(len(new_slot_mappings), num_layers)
        self.assertEqual(len(new_block_tables), num_layers)

    def test_generate_draft_tokens_eos(self):
        """测试草稿模型生成EOS token后是否停止"""
        num_layers = len(self.main_model.transformer.h)
        _, _, slot_mappings, block_tables = self.block_manager.allocate_for_prefill(
            self.seq_id, num_layers, self.input_ids.shape[1]
        )
        
        # 强制草稿模型生成EOS（通过修改logits实现）
        def mock_draft_forward(*args, **kwargs):
            logits = torch.zeros(1, 1, self.config.vocab_size, device=self.device)
            logits[0, 0, self.eos_token_id] = 1e9  # 让EOS概率最大
            return (logits, None)
        
        # 替换草稿模型的forward方法
        self.draft_model.forward = mock_draft_forward
        
        # 生成候选token
        draft_tokens, _, _ = self.speculative_decoder._generate_draft_tokens(
            input_ids=self.input_ids,
            max_speculative_steps=3,
            block_manager=self.block_manager,
            seq_id=self.seq_id,
            slot_mappings=slot_mappings,
            block_tables=block_tables
        )
        
        # 验证只生成1个EOS token
        self.assertEqual(draft_tokens.shape[1], 1)
        self.assertEqual(draft_tokens[0, 0].item(), self.eos_token_id)

    def test_validate_draft_tokens_acceptance(self):
        """测试主模型验证候选token的接受逻辑"""
        # 预填充块管理器
        num_layers = len(self.main_model.transformer.h)
        _, _, slot_mappings, block_tables = self.block_manager.allocate_for_prefill(
            self.seq_id, num_layers, self.input_ids.shape[1]
        )
        
        # 生成3个候选token
        draft_tokens, new_slot_mappings, new_block_tables = self.speculative_decoder._generate_draft_tokens(
            input_ids=self.input_ids,
            max_speculative_steps=3,
            block_manager=self.block_manager,
            seq_id=self.seq_id,
            slot_mappings=slot_mappings,
            block_tables=block_tables
        )
        draft_len = draft_tokens.shape[1]
        self.assertGreater(draft_len, 0)
        
        # 主模型验证
        accepted_length, new_tokens = self.speculative_decoder._validate_draft_tokens(
            input_ids=self.input_ids,
            draft_tokens=draft_tokens,
            block_manager=self.block_manager,
            seq_id=self.seq_id,
            slot_mappings=new_slot_mappings,
            block_tables=new_block_tables
        )
        
        print(accepted_length,draft_len)
        accepted_length-=1
        # 验证接受长度在合理范围（0 ≤ accepted_length ≤ draft_len）
        self.assertGreaterEqual(accepted_length, 0)
        self.assertLessEqual(accepted_length, draft_len)
        
        # 验证new_tokens形状正确
        self.assertEqual(new_tokens.shape[1], draft_len)


if __name__ == "__main__":
    unittest.main()