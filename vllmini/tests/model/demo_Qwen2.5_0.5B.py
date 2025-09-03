import pprint
from modelscope import AutoTokenizer
from transformers import Qwen2Tokenizer,AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"

model_path = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

state_dict = model.state_dict()
state_str = pprint.pformat(state_dict.keys())

# 将内容写入当前目录的文本文件
with open("/home/xtc/project/vllmini/vllmini/tests/model/qwen2_0.5B_state_dict.txt", "w", encoding="utf-8") as f:
    f.write(state_str)



# tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
# prompt = "做个简单的自我介绍."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)