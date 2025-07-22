# vllmini-learning

**learning form https://github.com/MDK8888/vllmini** 

## LLM推理

**在LLM推理阶段 主要分为了两个阶段 Prefill和Decode阶段**

### Prefill阶段

  预填充阶段发生在模型接收到完整输入 Prompt 之后，但在开始生成第一个输出 token 之前。这个阶段的主要任务是处理输入的 Prompt，计算出所有输入 token 的上下文表示，并初始化后续解码阶段所需的数据结构 kvcache

   当输入一段Prompt，需要进行对应的Tokenization, 将prompt处理为对应的 token, 然后再通过 Embedding层将离散的token映射为对应的高维向量以及加入对应的编码，转换为模型可以处理的形式,对于prompt每个token 通过 Embedding 隐射的高维向量进行拼接为 $X$，然后进行对应后续的计算，通过和transfomer训练得来的 权重矩阵 $W_Q, W_K, W_V$ ，进行对应矩阵乘法运算，得到 prompt的 token 对应的 Q，K，V 矩阵

  接下来，计算注意力分数。对于每个注意力头，我们将 Query 和 Key 的转置相乘。


$$
AttentionScore = Q \cdot K^T
$$


然后，对注意力分数进行缩放（除以 $\sqrt{head\_dim}$） Softmax 归一化，得到注意力权重矩阵，最后，将注意力权重与 Value 矩阵相乘，得到自注意力层的输出：


$$
AttentionOutput = AttentionWeight \cdot V
$$


得到注意力层的输出以后，需要将输入序列的`k`和`v`存入 kv 缓存。之后注意力输出矩阵，会先经过前馈神经网络，然后经线性层 + softmax 预测下一个 token 的概率，模型会选择概率最高的 token 作为第一个输出。至此，从注意力输出矩阵到第一个 token 的预测完成。

  预填充阶段的一个关键优势在于其高度的并行性，由于整个输入 Prompt 在开始时是已知的，模型利用矩阵可以同时计算所有 token 在每一层的表示。这种处理方式使得即使在较小的batch size下也能使得GPU的利用率很高，如在prefill阶段需要处理长输入，则这个阶段的计算开销会比较大，显卡利用率很容易打满了。如果增大batch size时，prefill阶段每个token的处理开销几乎保持不变，这意味着prefill的效率在小batch size时就已经非常高，说明开销是一定的。

### Decode阶段

  解码阶段在预填充阶段完成之后开始。在这个阶段，模型以自回归的方式逐个生成输出 token。每生成一个 token，该 token 就会被添加到已生成的序列中，并作为下一步生成的输入。

  在Decode阶段大部分的执行的操作和Prefill阶段有相同的地方，但不同的是prefill可以并行处理多个token和计算相关的注意力矩阵的，Decode阶段则 以自回归的方式 把上一次生成的token 添加到之前的序列作为输入进行逐个生成 token,在这一阶段为了提升对应的速度， 会充分利用之前在 prefill阶段生成的 KV缓存吗，同时在之后的阶段 添加和维护这个KV缓存。

  首先decode阶段会接受来自上一层生成token，由于之前已经缓存了之前序列对应的k,v向量，在此阶段只需要 计算该token对应  Q，K，V 向量，将当前生成 token 的 Key 和 Value 向量更新到 KV 缓存中。在自注意力计算中，当前 token 的 Query 向量会与 KV 缓存中所有历史 token 的 Key 向量进行比较，计算注意力权重。Value 向量会根据这些权重进行加权求和，得到上下文向量。模型最后一层的输出会经过线性层和 Softmax 函数，得到下一个 token 的概率分布。根据解码策略从概率分布中选择下一个 token。

### **性能瓶颈**

- **预填充阶段：** 对于极长的输入 Prompt，自注意力机制的计算量仍然很大，可能成为计算瓶颈。同时，加载模型权重和初始 KV 缓存的内存开销也需要考虑。
- **解码阶段：** 解码的串行自回归特性是主要的瓶颈。虽然 KV 缓存减少了重复计算，但每一步仍然需要进行注意力计算，并且随着生成序列的增长，KV 缓存的大小也会增加，可能导致内存带宽瓶颈。此外，生成长序列会显著增加总的推理时间。

## vllmini

### 项目的架构图
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/24688814debc4c3ab1b3cd34033c4e73~tplv-k3u1fbpfcp-watermark.image?)

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

​	再下一个阶段则是向block_manage 进行 decode资源的申请，并调用模型进行forward()推理，得到推理所得到token的logits,提取并保存最后一个在本地的哈希字典中，接着会进行一些验证，检查是否推理已经完成生成（遇到EOS或达到最大长度），如已经完成，则调用 `remove_sequence_from_processing()` 对本地维护哈希字典进行一些kv值的删除，再通过block_manage进行申请资源的释放，如未完成，则进行另外的逻辑的处理，首先schedualer 会在服务的整个生命周期进行运行run()函数，然后进行会不断的检查 优先队列是否有对应的序列需要处理，所以如果推理还未完成，则会重新加入到优先队列中，由于python的 优先队列的数据结构实现是由小顶堆实现的，因为序列的id是请求到达时间戳所生成的，同时代码中进行队列put()操作，所使用的比较优先级的元素则使用了时间戳，根据小顶堆的特性，整体的任务调度实现有点类似操作系统任务调度的“先来先服务”的调度策略，在整个任务调度器 调度策略方面实现稍微单调了一些，在真正实际的可能得考虑多方面的优化，比如整个调度器主要的实现的是单线程的不断轮询查找队列是否需要处理，如果轮询时间设置不当，很容易造成性能的浪费，同时如果先来先服务的话，意味着不把前一个时间戳 序列任务处理完成的话，后续的请求推理都不会被执行，当某个时间戳序列任务 因为某些情况迟迟无法完成，可能导致后续系统无法正常运行，在之前prefill申请的资源也无法释放，可能导致内存泄漏的风险，当然了，这是是整个系统设计方面的事情了，而且项目的Redeme文档就说明了，项目主要的初衷是学习为主。

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



