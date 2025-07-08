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

