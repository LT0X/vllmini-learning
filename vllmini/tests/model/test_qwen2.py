#pytest vllmini/tests/model/test_qwen2.py::TestQwen2WithPagedAttention::test_prefill_and_decode_one_token -v -s
# 获取当前文件所在目录
import os
import sys

current_file_path = os.path.abspath(__file__)
# 计算项目根目录（根据你的路径结构需要向上退3级）
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
print(project_root)
# 将项目根目录添加到Python的搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import unittest
from vllmini.model.qwen2 import Qwen2LMHeadModel  # 假设已实现Qwen2模型类
from transformers import AutoModelForCausalLM, Qwen2Tokenizer, Qwen2Config
import torch.nn.functional as F

def generate_triangular_mask(batch_size, num_heads, seq_len, dtype=torch.float16):
    # 创建上三角掩码，包含对角线，值为-inf
    upper_triangular = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), dtype=dtype),
        diagonal=1
    )
    
    # 扩展掩码形状以匹配需求
    mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # 形状: (batch_size, num_heads, seq_len, seq_len)
    
    return mask

class TestQwen2WithPagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 加载Qwen2配置和模型
        cls.model_name = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
        cls.main_model_name = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
        cls.config = Qwen2Config.from_pretrained(cls.model_name)
        # cls.main_config = Qwen2Config.from_pretrained(cls.main_model_name)
        print("加载模型和tokenizer成功")  
        cls.model = Qwen2LMHeadModel(cls.config)
        # cls.main_model = Qwen2LMHeadModel(cls.main_config)
        print("加载模型和tokenizer成功")
        cls.model.load_huggingface_weights(cls.model_name)
        # cls.main_model.load_huggingface_weights(cls.main_model_name)
        
        # 加载tokenizer并设置pad_token
        cls.tokenizer = Qwen2Tokenizer.from_pretrained(cls.model_name)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        # 设置设备
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)
        cls.model.eval()

    # def test_prefill_stage(self):
    #     input_text = "Hello, how are you?"
    #     input_ids = self.tokenizer.encode(
    #         input_text, 
    #         return_tensors="pt",
    #         padding=False,
    #         truncation=True
    #     ).to(self.device)
    #     attention_mask = torch.ones_like(input_ids)
    #     position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

    #     # 准备prefill阶段的缓存张量
    #     seq_len = input_ids.size(1)
    #     num_layers = self.config.num_hidden_layers
    #     num_heads = self.config.num_attention_heads
    #     head_size = self.config.hidden_size // num_heads
    #     # Qwen2的注意力块结构可能不同，根据实际实现调整
    #     block_size = self.model.model.layers[0].self_attn.block_size
    #     num_blocks = 1024

    #     # 初始化KV缓存
    #     key_cache = torch.zeros(
    #         num_blocks, num_heads, head_size // 8, block_size, 8,
    #         dtype=torch.float16, device=self.device
    #     )
    #     value_cache = torch.zeros(
    #         num_blocks, num_heads, head_size, block_size,
    #         dtype=torch.float16, device=self.device
    #     )
        
    #     # 准备slot mapping
    #     slot_mapping = []
    #     for i in range(num_layers):
    #         layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
    #         slot_mapping.append(layer_slot_mapping)

    #     with torch.no_grad():
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             position_ids=position_ids,
    #             attention_mask=attention_mask,
    #             use_cache=True,
    #             is_prefill=True,
    #             key_cache=key_cache,
    #             value_cache=value_cache,
    #             slot_mappings=slot_mapping
    #         )

    #     # 验证输出
        
    #     print(outputs[0])
    #     self.assertIsNotNone(outputs[0])  # 检查是否生成logits
    #     self.assertIsNotNone(outputs[1])  # 检查是否生成KV缓存
    #     self.assertEqual(len(outputs[1]), self.config.num_hidden_layers)  # 检查所有层都生成缓存
        
    #     # 检查缓存是否被填充
    #     self.assertFalse(torch.all(key_cache == 0))
    #     self.assertFalse(torch.all(value_cache == 0))
    
    def test_prefill_and_decode_one_token(self):
        # input_text = "Hello, how are"
        input_text = "你好，我是小明,"
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=True
        ).to(self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        # 准备缓存张量
        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        # block_size = self.model.model.layers[0].self_attn.block_size
        block_size = self.model.model.layers[0].self_attn.block_size
        num_blocks = 1024
        print("block_size:", block_size)
        # 生成注意力掩码
        attention_mask = generate_triangular_mask(1, num_heads, seq_len)
        attention_mask = attention_mask.to(self.device)
     #    attention_mask = torch.ones_like(input_ids)
        # 初始化KV缓存
        key_cache = torch.zeros(
            num_blocks, num_heads, head_size // 8, block_size, 8,
            dtype=torch.float16, device=self.device
        )
        value_cache = torch.zeros(
            num_blocks, num_heads, head_size, block_size,
            dtype=torch.float16, device=self.device
        )
        
        # Prefill阶段
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        with torch.no_grad():
            prefill_outputs, kv_cache = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping
            )
        
        # main_outputs = self.main_modle_prefill()
        
        # print(kv_cache)
        
        # 验证prefill输出
        self.assertIsNotNone(prefill_outputs[0])
        self.assertIsNotNone(kv_cache)
        self.assertEqual(len(kv_cache), self.config.num_hidden_layers)
        self.assertFalse(torch.all(key_cache == 0))
        self.assertFalse(torch.all(value_cache == 0))

        


       
        # 从最后一个logits采样下一个token
        last_token_logits = prefill_outputs[0, -1, :]

        logits = last_token_logits
        temperature = 1.0
        logits = logits / temperature
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[next_token_index[0]]
        
        xx = F.softmax(logits, dim=-1)
        print("shape of logits"+str(xx.shape))
        print(next_token)
        tokens = 101360
        print("概率"+str(xx[tokens]))
        # probs = torch.softmax(last_token_logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)
        print("prefill 生成的token",str(self.tokenizer.decode(next_token)))
        generated_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        print("prefill 生成的token",str(self.tokenizer.decode(next_token)))
        # 解码阶段（一个token）
        current_length = input_ids.size(1)
        position_ids = torch.tensor([current_length], dtype=torch.long, device=self.device)
        
        # 准备解码阶段的slot mapping
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.tensor([seq_len], dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        # 准备block tables
        block_tables = []
        for i in range(num_layers):
            block_table = torch.tensor([i, -1, -1, -1]).unsqueeze(0).to(dtype=torch.int32, device=self.device)
            block_tables.append(block_table)

        # 准备序列长度
        seq_lens = torch.tensor([current_length], dtype=torch.int32, device=self.device)
        # key_cache, value_cache = kv_cache[0], kv_cache[1]
        non_zero_mask = key_cache != 0
        non_zero_values = key_cache[non_zero_mask]
        print(non_zero_values)
        
        with torch.no_grad():
            decode_outputs = self.model(
                input_ids=next_token.unsqueeze(0).unsqueeze(0),
                position_ids=position_ids,
                attention_mask=None,
                use_cache=True,
                is_prefill=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=block_size
            )


        last_token_logits = decode_outputs[0][0]

        logits = last_token_logits
        temperature = 1.0
        logits = logits / temperature
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[0, next_token_index[0]]

        # probs = torch.softmax(last_token_logits, dim=-1)
        # next_token = torch.argmax(probs, dim=-1)
        # print("概率为："+str(probs[0][next_token]))
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        # 验证解码是否生成了一个新token
        self.assertEqual(generated_ids.size(1), input_ids.size(1) + 2)

        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        token = self.tokenizer.decode(next_token)
        print(f"Input text: {input_text}")
        print(f"Generated text: {generated_text}")
        print(f"decode Generated token: {token}")

        # 额外检查
        self.assertNotEqual(input_text, generated_text)  # 确保生成了内容
        self.assertTrue(generated_text.startswith(input_text))  # 确保生成内容以输入为开头
    def main_modle_prefill(self,):
        nput_text = "你好，请问你还"
        input_ids = self.tokenizer.encode(
            nput_text,
            return_tensors="pt",
            padding=False,
            truncation=True
        ).to(self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)
        self.main_model.to(self.device)
        # 准备缓存张量
        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        block_size = self.model.model.layers[0].self_attn.block_size
        num_blocks = 1024

        # 生成注意力掩码
        attention_mask = generate_triangular_mask(1, num_heads, seq_len)
        attention_mask = attention_mask.to(self.device)
        # attention_mask = torch.ones_like(input_ids)
        # 初始化KV缓存
        key_cache = torch.zeros(
            num_blocks, num_heads, head_size // 8, block_size, 8,
            dtype=torch.float16, device=self.device
        )
        value_cache = torch.zeros(
            num_blocks, num_heads, head_size, block_size,
            dtype=torch.float16, device=self.device
        )
        
        # Prefill阶段
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        with torch.no_grad():
            prefill_outputs, kv_cache = self.main_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping
            )
      
        return prefill_outputs

   

if __name__ == '__main__':
    unittest.main()