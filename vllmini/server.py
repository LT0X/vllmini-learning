import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # 用于请求和响应的数据验证
from transformers import GPT2Config, GPT2Tokenizer  # GPT2的配置和分词器
from transformers import AutoTokenizer
import torch

# 导入自定义的调度器、块管理器和GPT2模型实现
from vllmini.scheduler import Scheduler
from vllmini.block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from vllmini.speculative_decoding import LLMSpeculativeDecoding
from vllmini.model.qwen_v1 import QwenLMHeadModel


# 定义请求数据模型：接收用户输入的提示词和最大生成长度
class GenerationRequest(BaseModel):
    prompt: str  # 用户输入的文本提示
    max_length: int = 64  # 可选参数，生成文本的最大长度，默认64


# 定义生成请求的响应模型：返回序列ID（用于后续查询结果）
class GenerationResponse(BaseModel):
    sequence_id: int  # 生成任务的唯一标识ID


# 定义结果查询的响应模型：返回生成状态和结果
class ResultResponse(BaseModel):
    status: str  # 状态："in progress"（处理中）、"completed"（完成）、"error"（错误）
    generated: str = None  # 生成的文本，仅当状态为completed时有值


# 全局变量：将在服务启动时初始化
scheduler = None  # 调度器实例，管理所有生成任务
tokenizer = None  # GPT2分词器，用于文本与token的转换
qwen_tokenizer = None #Qwen分词器，用于文本与token的转换
device = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备（优先使用GPU）


#对于初始化投机解码组件封装函数，尽可能避免在原组件直接添加代码
def init_LLMSpeculativeDecoding()->LLMSpeculativeDecoding:
    
    global qwen_tokenizer
    main_model_path = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen-7B-Chat-Int4"
    draft_model_path = "/home/xtc/.cache/modelscope/hub/models/Qwen/Qwen-1_8B-Chat-Int4"
    
    main_model = QwenLMHeadModel.from_pretrained(main_model_path)
    draft_model = QwenLMHeadModel.from_pretrained(draft_model_path)
    
    qwen_tokenizer = AutoTokenizer.from_pretrained(main_model_path)

    main_block_manager = BlockManager(
        num_blocks=1000,
        block_size=16,
        num_heads=config.num_attention_heads,
        head_size=config.hidden_size // config.num_attention_heads,
        max_blocks_per_seq=4
    )

    draft_block_manager = BlockManager(
        num_blocks=1000,
        block_size=16,
        num_heads=config.num_attention_heads,
        head_size=config.hidden_size // config.num_attention_heads,
        max_blocks_per_seq=4
    )

    return LLMSpeculativeDecoding(
        main_model= main_model,
        draft_model= draft_model,
        main_block_manager= main_block_manager,
        draft_block_manager= draft_block_manager,
        tokenizer=tokenizer,
        max_speculative_steps=3,
        eos_token_id=50256
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI的生命周期管理器：负责服务启动时的初始化和关闭时的资源清理
    """
    # 服务启动时执行
    global scheduler, tokenizer, device

    #初始化投机解码组件（all）
    speculative_decoding = init_LLMSpeculativeDecoding()
    
    
    # 初始化组件
    config = GPT2Config.from_pretrained("gpt2")  # 加载GPT2的配置
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 加载GPT2分词器
    
    # 配置KV缓存块管理器的参数
    num_blocks = 1000  # 总缓存块数量
    num_heads = config.num_attention_heads  # 注意力头数量（从模型配置获取）
    head_size = config.hidden_size // num_heads  # 每个注意力头的维度
    block_size = 16  # 每个缓存块的大小（可存储的token数量）
    max_blocks_per_seq = 4  # 每个序列最多可使用的缓存块数量
    
    # 初始化块管理器（负责KV缓存的分配和释放）
    block_manager = BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        max_blocks_per_seq=max_blocks_per_seq
    )

    
    # 初始化GPT2模型并加载预训练权重
    model = GPT2LMHeadModel(config)
    model.load_huggingface_weights("gpt2")  # 加载HuggingFace格式的预训练权重
    model = model.to(device).to(torch.float16)  # 移动到指定设备并使用半精度（节省内存）
    
    # 初始化调度器（管理生成任务队列和执行）
    scheduler = Scheduler(
        model=model,
        block_manager=block_manager,
        speculative_decoding=speculative_decoding,
        max_length=20  # 序列的最大生成长度（可根据需求调整）
    )
    
    # 在后台任务中启动调度器
    scheduler_task = asyncio.create_task(run_scheduler())
    
    yield  # 服务运行期间保持此状态
    
    # 服务关闭时执行：清理资源
    scheduler_task.cancel()  # 取消调度器后台任务
    try:
        await scheduler_task  # 等待任务取消完成
    except asyncio.CancelledError:
        pass  # 捕获取消异常，忽略处理


async def run_scheduler():
    """
    异步运行调度器的后台任务：循环处理生成队列中的任务
    """
    while True:
        scheduler.run()  # 执行调度器的主循环（处理队列中的序列）
        await asyncio.sleep(0.01)  # 短暂休眠，避免过度占用CPU


# 创建FastAPI应用实例，指定生命周期管理器
app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    接收文本生成请求的API端点：将用户输入的提示词转为生成任务并返回序列ID
    """
    # 声明全局变量，以便在函数内部使用
    global scheduler, tokenizer, device
    
    # 检查调度器是否已初始化（未初始化则返回服务不可用）
    if scheduler is None:
        raise HTTPException(status_code=503, detail="调度器尚未初始化")

    # 将用户输入的提示词编码为token ID（模型可处理的数字形式）
    tokens = tokenizer.encode(request.prompt)
    input_ids = torch.tensor([tokens], dtype=torch.int64, device=device)  # 转换为张量并移动到指定设备

    # 将输入添加到调度器，获取任务唯一ID
    seq_id = scheduler.add_sequence(input_ids)

    # 返回生成任务的ID（用于后续查询结果）
    return GenerationResponse(sequence_id=seq_id)


@app.get("/result/{seq_id}", response_model=ResultResponse)
async def get_result(seq_id: int):
    """
    查询生成结果的API端点：根据序列ID返回当前生成状态和结果
    """
    global scheduler, tokenizer
    
    # 检查调度器是否已初始化
    if scheduler is None:
        raise HTTPException(status_code=503, detail="调度器尚未初始化")

    # 检查序列ID是否存在于生成任务中
    if seq_id in scheduler.sequences:
        # 获取生成的token ID序列并解码为文本
        generated_ids = scheduler.sequences[seq_id]
        generated_tokens = tokenizer.decode(generated_ids[0].tolist())

        # 判断任务状态：活跃（处理中）或已完成
        if seq_id in scheduler.active_sequences:
            return ResultResponse(status="in progress", generated=generated_tokens)
        
        # 任务完成后，移除已完成的序列数据并返回结果
        scheduler.remove_completed_sequence(seq_id)
        return ResultResponse(status="completed", generated=generated_tokens)

    # 若序列ID不存在，返回错误状态
    return ResultResponse(status="error")

@app.post("/generatePro", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    投机解码版本：将用户输入的提示词转为生成任务并返回序列ID
    """
    # 声明全局变量，以便在函数内部使用
    global scheduler, qwen_tokenizer, device
    
    # 检查调度器是否已初始化（未初始化则返回服务不可用）
    if scheduler is None:
        raise HTTPException(status_code=503, detail="调度器尚未初始化")

    # 将用户输入的提示词编码为token ID（模型可处理的数字形式）
    tokens = qwen_tokenizer.encode(request.prompt)
    input_ids = torch.tensor([tokens], dtype=torch.int64, device=device)  # 转换为张量并移动到指定设备

    # 将输入添加到调度器，获取任务唯一ID
    seq_id = scheduler.add_sequence_pro(input_ids)

    # 返回生成任务的ID（用于后续查询结果）
    return GenerationResponse(sequence_id=seq_id)

@app.get("/resultPro/{seq_id}", response_model=ResultResponse)
async def get_result(seq_id: int):
    """
    查询生成结果的API端点：根据序列ID返回当前生成状态和结果
    """
    global scheduler, tokenizer
    
    # 检查调度器是否已初始化
    if scheduler is None:
        raise HTTPException(status_code=503, detail="调度器尚未初始化")

    # 检查序列ID是否存在于生成任务中
    if seq_id in scheduler.sequences:
        # 获取生成的token ID序列并解码为文本
        generated_ids = scheduler.sequences[seq_id]
        generated_tokens = tokenizer.decode(generated_ids[0].tolist())

        # 判断任务状态：活跃（处理中）或已完成
        if seq_id in scheduler.active_sequences:
            return ResultResponse(status="in progress", generated=generated_tokens)
        
        # 任务完成后，移除已完成的序列数据并返回结果
        scheduler.remove_completed_sequence(seq_id)
        return ResultResponse(status="completed", generated=generated_tokens)

    # 若序列ID不存在，返回错误状态
    return ResultResponse(status="error")
                                                                             
# 当脚本直接运行时，启动UVicorn服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 监听所有网络接口的8000端口