import unittest
import torch
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config
from vllmini.block_manager import BlockManager
from vllmini.speculative_decoding import LLMSpeculativeDecoding  # 替换为实际模块名

class TestLLMSpeculativeDecoding(unittest.TestCase):
    def setUp(self):
        # 初始化模型和tokenizer
        self.config = GPT2Config.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 主模型和草稿模型
        self.main_model = GPT2LMHeadModel(self.config)
        self.draft_model = GPT2LMHeadModel(self.config)
        self.main_model.load_huggingface_weights("gpt2")
        self.draft_model.load_huggingface_weights("gpt2")
        
        # 移动到合适的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_model = self.main_model.to(self.device).to(torch.float16)
        self.draft_model = self.draft_model.to(self.device).to(torch.float16)
        
        # 初始化block manager
        self.num_blocks = 100
        self.block_size = 16
        self.num_heads = self.config.num_attention_heads
        self.head_size = self.config.hidden_size // self.num_heads
        self.max_blocks_per_seq = 4
        self.block_manager = BlockManager(
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_size,
            self.max_blocks_per_seq
        )
        
        # 初始化投机解码器
        self.speculative_decoder = LLMSpeculativeDecoding(
            main_model=self.main_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            max_speculative_steps=3,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 测试输入
        self.prompt = "Hello, world!"
        self.input_ids = self.tokenizer.encode(
            self.prompt, 
            return_tensors="pt"
        ).to(self.device)
        self.max_length = 30
        self.seq_id = 1  # 新增序列ID

    def test_generate_basic_functionality(self):
        """测试生成功能是否正常工作"""
        with torch.no_grad():
            generated = self.speculative_decoder.generate(
                input_ids=self.input_ids,
                max_length=self.max_length,
                block_manager=self.block_manager,
                seq_id=self.seq_id  # 传递seq_id
            )
        
        # 检查生成结果的基本属性
        self.assertIsInstance(generated, torch.Tensor)
        self.assertEqual(generated.shape[0], 1)  # 批次大小为1
        self.assertLessEqual(generated.shape[1], self.max_length)  # 不超过最大长度
        self.assertGreater(generated.shape[1], self.input_ids.shape[1])  # 有新内容生成
        
        # 检查生成结果是否包含输入前缀
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(self.input_ids[0], skip_special_tokens=True)
        self.assertTrue(generated_text.startswith(input_text))

    def test_eos_handling(self):
        """测试是否能正确处理EOS token"""
        # 使用可能快速生成EOS的短提示
        short_prompt = "End here."
        input_ids = self.tokenizer.encode(short_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated = self.speculative_decoder.generate(
                input_ids=input_ids,
                max_length=50,
                block_manager=self.block_manager,
                seq_id=2  # 新的序列ID
            )
        
        # 检查是否包含EOS token（如果生成了的话）
        generated_ids = generated[0].tolist()
        if self.tokenizer.eos_token_id in generated_ids:
            eos_pos = generated_ids.index(self.tokenizer.eos_token_id)
            self.assertEqual(len(generated_ids), eos_pos + 1)  # EOS是最后一个token


if __name__ == "__main__":
    unittest.main()