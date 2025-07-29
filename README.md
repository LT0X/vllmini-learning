# vllmini-learning

**learning form https://github.com/MDK8888/vllmini** 

## LLM推理

**在LLM推理阶段 主要分为了两个阶段 Prefill和Decode阶段**

### Prefill阶段

​	预填充阶段发生在模型接收到完整输入 Prompt 之后，但在开始生成第一个输出 token 之前。这个阶段的主要任务是处理输入的 Prompt，计算出所有输入 token 的上下文表示，并初始化后续解码阶段所需的数据结构 kvcache

​	当输入一段Prompt，需要进行对应的Tokenization, 将prompt处理为对应的 token, 然后再通过 Embedding层将离散的token映射为对应的高维向量以及加入对应的编码，转换为模型可以处理的形式,对于prompt每个token 通过 Embedding 隐射的高维向量进行拼接为 $X$，然后进行对应后续的计算，通过和transfomer训练得来的 权重矩阵 $W_Q, W_K, W_V$ ，进行对应矩阵乘法运算，得到 prompt的 token 对应的 Q，K，V 矩阵

​	 接下来，计算注意力分数。对于每个注意力头，我们将 Query 和 Key 的转置相乘。


$$
AttentionScore = Q \cdot K^T
$$


​	然后，对注意力分数进行缩放（除以 $\sqrt{head\_dim}$） Softmax 归一化，得到注意力权重矩阵，最后，将注意力权重与 Value 矩阵相乘，得到自注意力层的输出：


$$
AttentionOutput = AttentionWeight \cdot V
$$

​	得到注意力层的输出以后，需要将输入序列的`k`和`v`存入 kv 缓存。之后注意力输出矩阵，会先经过前馈神经网络，然后经线性层 + softmax 预测下一个 token 的概率，模型会选择概率最高的 token 作为第一个输出。至此，从注意力输出矩阵到第一个 token 的预测完成。

​	预填充阶段的一个关键优势在于其高度的并行性，由于整个输入 Prompt 在开始时是已知的，模型利用矩阵可以同时计算所有 token 在每一层的表示。这种处理方式使得即使在较小的batch size下也能使得GPU的利用率很高，如在prefill阶段需要处理长输入，则这个阶段的计算开销会比较大，显卡利用率很容易打满了。如果增大batch size时，prefill阶段每个token的处理开销几乎保持不变，这意味着prefill的效率在小batch size时就已经非常高，说明开销是一定的。

### Decode阶段

​	 解码阶段在预填充阶段完成之后开始。在这个阶段，模型以自回归的方式逐个生成输出 token。每生成一个 token，该 token 就会被添加到已生成的序列中，并作为下一步生成的输入。

​	在Decode阶段大部分的执行的操作和Prefill阶段有相同的地方，但不同的是prefill可以并行处理多个token和计算相关的注意力矩阵的，Decode阶段则 以自回归的方式 把上一次生成的token 添加到之前的序列作为输入进行逐个生成 token,在这一阶段为了提升对应的速度， 会充分利用之前在 prefill阶段生成的 KV缓存吗，同时在之后的阶段 添加和维护这个KV缓存。

​	首先decode阶段会接受来自上一层生成token，由于之前已经缓存了之前序列对应的k,v向量，在此阶段只需要 计算该token对应  Q，K，V 向量，将当前生成 token 的 Key 和 Value 向量更新到 KV 缓存中。在自注意力计算中，当前 token 的 Query 向量会与 KV 缓存中所有历史 token 的 Key 向量进行比较，计算注意力权重。Value 向量会根据这些权重进行加权求和，得到上下文向量。模型最后一层的输出会经过线性层和 Softmax 函数，得到下一个 token 的概率分布。根据解码策略从概率分布中选择下一个 token。

### **性能瓶颈**

- **预填充阶段：** 对于极长的输入 Prompt，自注意力机制的计算量仍然很大，可能成为计算瓶颈。同时，加载模型权重和初始 KV 缓存的内存开销也需要考虑。
- **解码阶段：** 解码的串行自回归特性是主要的瓶颈。虽然 KV 缓存减少了重复计算，但每一步仍然需要进行注意力计算，并且随着生成序列的增长，KV 缓存的大小也会增加，可能导致内存带宽瓶颈。此外，生成长序列会显著增加总的推理时间。

## vllmini

### 项目的架构图

![123](https://i-blog.csdnimg.cn/direct/ad1eec6c90074183a803ef5dcc090ee5.png)



### server

​	首选项目实现了大模型推理的主要功能，主要的呈现方式以server/client为主，server方面实现了两个接口，在整个服务启动期间，会初始化所需组件，如  `block_manager`以及`scheduler`和加载对应的模型和参数，实现的两个接口分别为`/generate`和`/result/`

```python
`@app.post("/generate", response_model=GenerationResponse)`
`async def generate(request: GenerationRequest):`
```

这个接口的功能主要是，首先解析用户发送的提示词相关请求参数，然后发送提示词和请求参数发送给 `schedualer` ，有在`schedualer`接受到请求后，然后会由它创建对应的序列，然后进行相关的处理并提交任务在本地，然后会直接返回一个唯一序列id给用户，用户可根据这个id 去查询模型推理的结果，主要是以一种异步的方式来获取结果。

```python
@app.get("/result/{seq_id}", response_model=ResultResponse)

async def get_result(seq_id: int):
```

​	另外的一个接口实现的则是根据 `/generate`生成的seq_id来查询对应的模型推理结果，首先会查看 根据id来查看 任务的处理情况，如果已经完成，则返回对应的结果，如为其他的一种状态则 返回状态处理的信息。

### schedualer

​	主要是充当整个项目的任务的调度器，在server启动的时候，会对其进行初始化并实例化，然后会在服务的整个生命周期存在，同时会在后台一直运行`run()` 函数来处理推理任务。

```python
async def run_scheduler():
  """ 异步运行调度器的后台任务：循环处理生成队列中的任务 """
  while True:
​    scheduler.run()  # 执行调度器的主循环（处理队列中的序列）
​    await asyncio.sleep(0.01)  # 短暂休眠，避免过度占用CPU
```

​	在接口`/generate`  接受到客户端prompt参数和相关参数的时候，server会调用 schedualer组件的 add_sequence()函数，进行对应的处理以此来添加新的任务序列，首先会生成 请求序列的时间戳，以及唯一的id,  并且添加本地维护的优先队列的数据结构中，充当着调度器的任务队列，以便run()函数可以根据队列来处理请求，紧接着进行大模型推理的 prefill阶段，在prefill阶段之前，需要通过向block_manager 组件进行申请 所需的资源，获得申请完成资源后，对prefill阶段还需要进行掩码操作，因为prefill阶段的时候，全部的之前的prompt是已经知道的，但是在自注意力机制里的时候，我们希望保证模型在预测某个位置的输出时，只能 看到该位置之前的输入，而无法获取未来位置的信息，因果掩码的作用就相当于模型在预测下一个词的时候，只能依据已经出现的前文内容，主要的实现形式则是 一个三角矩阵，主对角线以及对角线下方的元素都被设置为0,对角线右上方的元素则被设为负无穷，在后续的softamx（）计算，加上注意力矩阵，得到屏蔽的效果。

​	紧接着开始调用模型和传输参数进行prefill 阶段，prefill阶段完成后，得到 prefill 阶段推理的 logits, 因为是prefill 阶段，所以得到的  是通过 promopt 所有token 预测的下一个token,所以要通过 `self.last_logits[seq_id] = logits[:, -1, :]`  提取logits 最后一个token的预测结果，然后更新 schedual 维护的各种哈希表,当然在python是称为字典，理所当然的使用了 唯一的序列id 作为哈希表的key最后返回 生成的 序列唯一id。

```python
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
```

​	在整个服务生命周期，会在后台一直运行，用于处理已经提交到优先队列中的序列，首先会查询队列是否为空，不为空则在从优先队列中提取一个id，然后接着在会进行一系列的安全检查，如序列长度以及序列活跃度检测。 对于状态不符合和格式 不符合接下来的处理的的会直接返回。

​	验证通过则进行decode阶段的处理，首先通过 `next_token = self.sample_next_token(seq_id) `获取 前一次推理所得到的next_token,函数主要通过读取本地维护的哈希字典last_logits，得到预测最后一个token的logits,然后进行 temperature运算进行温度缩放，原始代码中temperature = 1.0 则保持原始分布，再通过top-k 选取前50个概率最高的分布，然后再通过soft-max进行转换概率分布，并通过multinomial()对概率分布随机的选取对应的目标token的索引，并返回对应的结果得到最终的next_token。

​	再下一个阶段则是向block_manage 进行 decode资源的申请，并调用模型进行forward()推理，得到推理所得到token的logits,提取并保存最后一个在本地的哈希字典中，接着会进行一些验证，检查是否推理已经完成生成（遇到EOS或达到最大长度），如已经完成，则调用 `remove_sequence_from_processing()` 对本地维护哈希字典进行一些kv值的删除，再通过block_manage进行申请资源的释放，如未完成，则进行另外的逻辑的处理，首先schedualer 会在服务的整个生命周期进行运行run()函数，然后进行会不断的检查 优先队列是否有对应的序列需要处理，所以如果推理还未完成，则会重新加入到优先队列中，由于python的 优先队列的数据结构实现是由小顶堆实现的，因为序列的id是请求到达时间戳所生成的，同时代码中进行队列put()操作，所使用的比较优先级的元素则使用了时间戳，根据小顶堆的特性，整体的任务调度实现有点类似操作系统任务调度的“先来先服务”的调度策略，在整个任务调度器 调度策略方面实现稍微单调了一些，在真正实际的可能得考虑多方面的优化，比如整个调度器主要的实现的是单线程的不断轮询查找队列是否需要处理，如果轮询时间设置不当，很容易造成性能的浪费，同时如果先来先服务的话，意味着不把前一个时间戳 序列任务处理完成的话，后续的请求推理都不会被执行，当某个时间戳序列任务 因为某些情况迟迟无法完成，可能导致后续系统无法正常运行，在之前prefill申请的资源也无法释放，可能导致内存泄漏的风险，虽然程序有异常捕抓机制检测资源的分配的问题，但不能完全解决问题。当然了，这是是整个系统设计方面的事情了，而且项目的Readme文档就说明了，项目主要的初衷是学习为主。

```python
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
```

​	在整个run() 运行期间会通过try关键字捕抓异常，捕抓的异常识别通用异常为主，当出现异常后，进行异常处理情况，为申请资源失败的情况，则是cuda 显存不足，处理为调用handle_out_of_memory()函数处理资源分配问题，按照我的理解应该是，代码应该移除最老的序列id,以持来为最新的序列id让出显存资源，但是实现中 是运用了 max()函数，先通过生成器表达式筛选不包含本次序列id的id列表,然后在通过key=self.active_sequences.get，作为比较的依据，获取最大的时间戳的 序列的id，这样的话，因为max的原因就变成了优先删除较处理序列id最新的序列任务，并不是我理解优先释放最老的序列任务的资源，来为新序列任务让出资源，这里我并没有理解作者这样做的原因。

```python
   def handle_out_of_memory(self, batch_seq_ids: List[int]):
        """处理内存不足的情况，这里按理说应该是移除最老的序列以释放内存，
        但代码的实现是移除了最新的时间戳序列id,那移除的是最新的序列任务了
        """
        print("Handling out of memory")
        if self.active_sequences:
            
            seq_to_remove = max(
                (seq for seq in self.active_sequences if seq not in batch_seq_ids),
                key=c,
                default=None
            )
            if seq_to_remove is None:
                seq_to_remove = max(self.active_sequences, key=self.active_sequences.get)
            print(f"Removing sequence {seq_to_remove} due to memory constraints")
            self.remove_sequence_from_processing(seq_to_remove)
        else:
            print("No active sequences to remove")

```

### KVCache

​	kvCache是一个里面的关键类，也是优化推理速度的一个重要机制，block_manager初始化后，会在其初始化中一并初始化KVCache类，在初始化KVCache，会根据模型对应的一些参数来进行初始化，并以此申请对应的kvcache资源，同时为了判断分配资源，为此在本地维护4个哈希字典。

```
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
```

​	KV Cache是一种为大模型推理加速技术。在大模型推理的逻辑是：根据当前轮输入tokens预测并输出下一个token，这两者拼接就得到了下一轮的输入，每一轮只比上一轮增加了一个token。这意味着当前轮包含了上一轮的部分计算。上一轮中每一层attention layer的key和value被当前轮复用，而不是重新计算，就能加速推理过程，这就是KV Cache的作用。

​	之前也说过大模型的推理被分为了Prefill和Decode两个阶段。简单来说，Prefill阶段将prompt喂给模型做forward计算，该过程一次性计算了每个prompt token对应的key和value，并且被缓存起来；在Decode阶段，模型每foward一次生成一个token。这两个阶段有较为明显的差异。

​	同时对于有多个级联的block的情况下，每一层都会保存上一轮的KV_cache以提供本轮的计算使用。

​	由于KVCache的作用，在模型进行预测下一个token的时候，只需要单独进行最新token的 Q，K，V向量，Q是当前 token 的查询向量，表示 “我需要关注哪些信息”,所以Q 仅与当前 token 相关，不依赖历史信息，因此**无需与之前的 Q 拼接**，**所以Q无需进行缓存**.

- - K 是所有 token 的 “索引标签”，表示 “我是哪个位置的信息”。
  - V 是所有 token 的 “内容载体”，包含实际的语义信息。

  所以Key和 Value需要累积拼接，从而形成全局上下文,从而为了加快计算速度，从而进行Key和Value进行缓存，也是一种空间换时间的思想。

​	当时在学习kv缓存的机制的时候，也有一些可行性的疑问，因为现有大模型往往有`N`个`(Transformer) Block`，每个block都有对应的自注意力块，第一层的kv缓存信息自不用说，但是从第二层的kv缓存信息，是由前一层的输出作为输入得到的，如果前一层的输出改变了，按理来说第二层及以后的缓存不会失效吗，当然了，大概率还是我理解的机制的理解问题，要不然kvcache也不会在业界大规模的应用，甚至是推理的一个标配，所以上网查询一些相关博客，刚好找到一个对应的[博客](https://blog.csdn.net/daihaoguang/article/details/141515660)有解释这方面的疑问，

​	如下图所示，刨除前面的b和np这两个维度，则当前轮Attention的形状为[sq, hn]，如下图中左侧的Attention部分所示；那么下一轮序列长度增加1，形状变成了[s+1, hn] = [sq', hn]。图中Attention蓝色部分表示当前轮需要计算的部分，而绿色部分则表示下一轮新添加的部分。

​	证明的可行性则只需要证明本轮和下一轮在`Attention`的蓝色部分计算结果是一致的。因为这部分一致保证了前后两轮在级联的`Block`中`attention layer`中`key`和`value`的前`sk`个也是一致的的，因为新的一轮也只是向sk的方向去增长。

​	按计算注意力的步骤，首先由Q和K^T的进行相乘，按矩阵乘法的计算规则，得到QK^T 在前一轮前sk列是一致，即是蓝色部分，然后就是进行掩码操作，由于对于每一个token只能看到之前的内容以此来预测，所以进行mask操作，表现的形式，则是对注意力分数矩阵进行上三角赋予一个-inf，再经过softmax得到概率分布数值表现为0，所以在与v矩阵相乘（注意是 Attention Prob * V）得到注意力矩阵的时候，根据矩阵相乘的计算规则，得到的注意力矩阵最后的与前一轮的蓝色部分是一致的。所以关键即是mask的操作，使得数据为0， 使得计算的时候，只能看到前面的value,而看不到后面的value。最终得到KVcache在多个block是有效的。

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2de9900b5134fe69caca33cc3ebe496~tplv-k3u1fbpfcp-watermark.image?)

#### page_attention

​	对于大模型的推理中，有一个需要面对的问题则是关于序列资源的分配的问题，也就是对显存进行预分配，一般来说，在大模型的推理中往往不知道token的生成数量，如果我们预分配资源为设定为最大值，就可能会造成以下的一些浪费。

- 预分配，但不会用到。就比如设定最大分配的大小为 500 token，但实际推理 到100token就已经结束了，那么之后分配的空间就已经得到浪费。
- 预分配，但尚为用到。如分配了最大的token为500，但实际推理的也刚好满足500token，但是当推理到前10个token，后面还有许多token没有用到，刚刚好已经没有对应的显存了，但同时后面的请求只需要生成几十个token就可完成任务，这时候会导致后续的任务无法继续运行。
- 这样的分配的制度，可能会导致许多的显存碎片，这些碎片无法满足后续的请求，导致资源浪费。

​	这些问题是不是有些熟悉，在操作系统中，我们学习的内存分配的时候，也有一些类似的问题，当时是通过引入逻辑地址和页管理机制来实现对进程内存的分配，page_attention 就是类似于操作系统的页表机制，引入page_attention来帮助解决序列分配现存的问题。

​	首先，page_attention 会把显存划分为以block单位，每个block可以存储若干个token，然后，类似于操作系统中页表机制以及虚拟内存，对于每一个请求，有逻辑kvcache和 物理显存kvcache，其中请求逻辑kvcache是连续的，而实际的存放token的物理显存kvcache可以不连续，每一个请求都有维护一个映射表来 映射逻辑kvcache和物理kvcache。从而达到对显存的高效分配，

​	page_attention可以做到按需分配，不提前分配，按block分配，以此减少碎片大小，同时使用逻辑kvcache，方便实现调用。

![](https://i-blog.csdnimg.cn/direct/a6285bf00e8945c580a77582f936ef69.png)

![](https://i-blog.csdnimg.cn/direct/d9c5f4ccb6394825acc9bc3184c36fee.png)

​	`allocate_prefill()`主要的作用是为大模型prefill推理阶段分配KVcache资源，首先的话，由于之前已经说明，在Transformer 模型中，每层都有自己独立的 KV 缓存。因此，对于一个序列，我们需要为每一层分配一个块来存储该层的 KV 值。这就是为什么需要分配`num_layers`个块的原因，接下来的话则是开始分配，具体的实现，是通过对本地维护的四个表进行更新，后续通过以此判断来调用kvcahce，`free_blocks`和`allocated_blocks` 的更新很容易理解，重点是后面两个表的更新，`block_tables`首先这个表的数据结构为`Dict[int, List[Tuple[int, int]]]`  是一个哈希表，key为请求序列id，value是一个存储着二元元组list的数据结构，主要存储的作用是分配的block块和有效长度，  `self.block_tables[seq_id] = [(block, min(seq_len, self.block_size)) for block in allocated]` 这个代码，遍历了所有分配的 allocatedd的block,然后通过`min(seq_len, self.block_size)` ，为每一个block进行有效长度的赋值，为什么使用的min()的原因是

- **当 `seq_len > block_size` 时**：序列需要被分割到多个块中，每个块的有效长度最大为 `block_size`。
- **当 `seq_len < block_size` 时**：序列可以完整放入一个块，但只需使用块的前 `seq_len` 个位置，剩余空间闲置。

​	这样的话可以很好的 **避免越界访问**，内存块按固定大小分配，但序列长度不一致。同时不会读取 / 写入超过块大小的位置（防止内存越界）。不会遗漏序列的任何 token（最后一个块可能不满，但仍会记录实际长度）。

​	同时下一个 `paged_attention_block_tables` ，首先其的数据结构为 `Dict[int, List[List[int]]]`同样是哈希表，但是这个是一个三层结构，依旧是用序列Id作为key，但是value代表的是 最外层代表序列id的所有层所分配的block,而里面的list代表的是每一层所分配的block,按我是这么理解的，但实际上在赋值的代码中，实际的类型应该是`Dict[int, List[torch.Tensor]]` 这就不是很清楚作者的意图了。

​	但总体表示的就是每一层所分配的物理块列表。然后的话`max_blocks_per_seq`定义了**每个层最多可以使用的块数量**。这是为了处理长序列的情况：当序列长度超过一个块的大小时，需要多个块来存储该层的 KV 缓存。`[[block] + [-1] * (self.max_blocks_per_seq - 1)]`表示一个层的块列表。初始时，每个层只分配一个块，因此列表中只有第一个位置有有效的块 ID，其余位置用`-1`填充。

​	接下来的话，就是slot_mappings了，这个的作用的本质上就是实现了分页注意力机制中逻辑位置到物理位置的映射，**遍历每个分配的物理块** (`for block in allocated`)，然后**生成逻辑位置索引** (`torch.arange(seq_len)`)：创建从 0 到`seq_len-1`的连续整数，表示 token 在序列中的逻辑位置，然后就是**计算物理块偏移量** (`block * self.block_size`)，通过确定当前物理块在全局 KV 缓存中的起始位置，最后进行**映射逻辑位置到物理位置**,将逻辑索引加上块偏移量，类似计算机组成原理里面的offset从而得到每个 token 在物理 KV 缓存中的实际位置。通过这样就可以使得逻辑地址到物理位置的映射。

​	python的语法有时候我觉得太自由了，语法糖挺多的，另外一个麻烦的点，因为是解释型和动态类型的编程语言，有时候的返回值和变量在运行的时候才知道类型（any类型），读项目源码和debug有时候还挺头疼的。

```python
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
    
```

​	`allocate_for_prefill`的作用为在prefill阶段请求序列初次分配kvcache资源,而`append_block`的作用则是在decode阶段在prefill分配的资源不过的时候，动态的添加kvcache资源，首先的话，查看是否有空闲块，获取空闲块号，然后更新块表以及填充token数量为0，这里有个注意的点，为什么为0，而prefill阶段为 `min(seq_len, self.block_size)`，`block_tables` 中每个元组 `(block_id, filled)` 的含义是：`block_id`：物理块的 ID，`filled`：该物理块中**已实际存储的 token 数量**（用于后续判断块是否已满，以及限制访问范围)。

- 在预填充阶段，我们的目标是**为输入序列的首次处理分配块，并立即填充该序列的 KV 缓存**。此时块中已经有实际数据，`filled` 需要记录 “这个块到底装了多少个 token”。
- 在动态扩展阶段，我们的目标是**新增一个空块，准备接收后续生成的 token**（比如解码阶段生成新 token 时，原块已满，需要新块存储）。此时新块刚分配，**还没有存储任何数据**，因此 `filled` 初始值为 `0`。

​	接下来更新`paged_attention_block_tables`,其实无非就是判断那一层的缺失block，找到值为-1的索引，然后对其值赋值为之前得到空闲的快号。

```python
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
```

