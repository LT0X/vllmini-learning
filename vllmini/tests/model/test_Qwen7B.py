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
state_str = pprint.pformat(state_dict.keys())


# 将内容写入当前目录的文本文件
with open("/home/xtc/project/vllmini/vllmini/tests/model/Qwen7B_state_dict.txt", "w", encoding="utf-8") as f:
    f.write(state_str)


# print(config)

# response, history = model.chat(tokenizer, "你知道", history=None)

# print(response)
# 你好！很高兴为你提供帮助。
