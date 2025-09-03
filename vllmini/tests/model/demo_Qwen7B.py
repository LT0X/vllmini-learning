import pprint
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig

# Note: The default behavior now has injection attack prevention off.
model_path = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen-7B-Chat-Int4"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)

state_dict = model.state_dict()
# 将有序字典转换为格式化的字符串
# state_str = pprint.pformat(state_dict.keys())

total_bytes = 0
for param in state_dict.values():
    # 每个元素的字节数 × 元素总数
    total_bytes += param.element_size() * param.numel()

# 转换为更易读的单位
def format_size(bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = bytes
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"
print(f"state_dict 总内存占用: {format_size(total_bytes)}")
# print(f"state_dict 总内存占用: {format_size(total_bytes)}")
# # 将内容写入当前目录的文本文件
# with open("/home/xtc/project/vllmini/vllmini/tests/model/Qwen7B_state_dict.txt", "w", encoding="utf-8") as f:
#     f.write(state_str)


# print(config)

# response, history = model.chat(tokenizer, "你知道", history=None)

# print(response)
# 你好！很高兴为你提供帮助。
