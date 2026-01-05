# LLM Background - LLM Infra: Deepspeed and vLLM

[TOC]



## 0. 总览

- **DeepSpeed**：偏「**训练/微调**」用的分布式加速框架，核心是 ZeRO 把大模型拆着放，省显存、提吞吐。
- **vLLM**：偏「**推理/服务**」用的高效推理引擎，核心是 PagedAttention 把 KV cache 管理好，提升 QPS、降低显存。
- 两者 **可以配合使用**：用 DeepSpeed 训练 / 微调完模型，然后用 vLLM 来部署推理服务；但一般不会在「同一个进程里」混搭。

<br>

## 1. DeepSpeed 是什么？

**DeepSpeed** 是***Microsoft***开源的一个深度学习训练优化库，主要干三件事：

1. **让大模型「塞得下」GPU：显存优化**
   - 普通 DataParallel：每张卡都放一份完整模型 + 优化器状态 + 梯度
   - 模型一旦上到 10B、几十 B，很快就 OOM
   - DeepSpeed 的 **ZeRO（Zero Redundancy Optimizer）** 把这些东西「切片」分布到多张卡上，消灭冗余
2. **让训练更快：并行 + 通信优化**
   - 支持 **数据并行 + 模型并行 +流水线并行 + ZeRO** 的混合；
   - 针对通信做了很多优化（overlap、压缩等）；
   - 可以结合 NVIDIA 的 NCCL、InfiniBand 做大集群训练。
3. **DS的实用的「训练工程」工具**
   - ZeRO-Stage 1/2/3
   - ZeRO-Offload（把某些状态放 CPU / NVMe）
   - 1-bit Adam / LAMB（通信压缩）
   - 和 HuggingFace Transformers、Megatron-LM、DeepSpeed-MII 等整合。

总结来说，DS最关键的是：

> **DeepSpeed = 一整套「省显存、能让大模型跑起来」的训练方案，特别是 ZeRO。**

<br>

## 2. ZeRO-2 和 ZeRO-3：到底差别在哪、什么时候用谁？

### 2.1 ZeRO 的基本思想

对于普通数据并行计算来说，通常情况下：

> 每张卡都存「整套」模型参数 + 梯度 + 优化器状态 → 典型“n 份冗余”。

**ZeRO 的核心：把这些大头切片（shard），每张卡只保存自己的一部分**，需要时再通过通信把必要的碎片聚起来算。

ZeRO 有三个主要 Stage：

- **Stage 1：shard 优化器状态**（optimizer states）
- **Stage 2：再 shard 梯度（gradients）**
- **Stage 3：连参数（model weights）都 shard 掉**

我们重点看 **ZeRO-2 vs ZeRO-3**。

### 2.2 ZeRO-2（Stage 2）—「中杯显存优化」

**做了什么？**

- Shard 优化器状态（Adam 的 moment 等）
- Shard 梯度
- **但每张卡仍然保留一份完整的模型权重**

这样一来：

- 显存中最吃的两块（optimizer states + grads）不再是 n 重复制
- 模型权重还是全量；前向/反向时「就地」计算，通信开销相对小

**优点：**

- 对代码侵入小：只要把优化器交给 DeepSpeed 管，前向/反向基本不变
- 通信开销相对 ZeRO-3 小
- 显存节省已经很可观
- 非常适合 **几十亿参数以内**、或者单卡本身显存比较大的情况（A100 80G / H100 80G）

**适用场景：**

- 模型 size：比如 7B–13B，这种单卡还能勉强塞下，只是 batch 不太大、seq 不太长
- 你有 **多张 GPU**，想把 batch 做大一点，又不想折腾太复杂
- 训练场景（SFT、RLHF 等）以 **稳定为主**：不想搞 CPU/NVMe offload 那种花活

你以后如果在 H100 上训 **Qwen3-8B / Qwen3-14B**，**ZeRO-2 是非常舒服的一档**：显存有余、逻辑简单、坑少。



### 2.3 ZeRO-3（Stage 3）—「大杯 / 急救型显存优化」

**做了什么？**

- 除了 ZeRO-2 的优化器状态 + 梯度切片
- **连模型参数本身也 shard 了**：
  - 每张卡只放一部分参数
  - 前向需要的时候，把相关 shard 聚合（all-gather），算完再散回去

进一步配合：

- **CPU offload**：部分参数/optimizer 状态搬到 CPU 内存，需要时搬回 GPU；
- 甚至可以 **NVMe offload**：极端情况下，上到几十 B、上百 B 的模型。

**优点：**

- 显存节省接近「极限」，是能让几十 B 级模型在有限 GPU 上「塞得下」的关键
- 配合 offload，可以让 40GB 卡也玩 30B+ 模型（当然会变慢）

**缺点：**

- 实现复杂：
  - 前向/反向时需要频繁 all-gather / reduce-scatter；
  - 要处理 offload 的异步通信、overlap，容易踩坑。
- 通信和 CPU I/O 开销大：
  - 如果你是单机少卡、没有高速网络，可能会感觉「怎么这么慢」。

**适用场景：**

- 大模型：20B、30B、65B、70B、100B…
- 单卡显存有限（40GB、48GB）但又想硬上 30B+ 模型
- 目标是「先跑起来再说」，速度不是第一优先级

结合到你这边：

- **14B 模型**：在 H100 80GB 上，用 ZeRO-2 就已经很舒服；
- **32B 模型**：
  - 如果只用单卡 80GB 且想全参微调，很大概率必须走 ZeRO-3 + offload；
  - 但这时候你会发现：**QLoRA + 4bit 量化 + LoRA Adapter 其实更优雅**。



### 2.4 总结一下「不同尺寸模型 → 用哪个 ZeRO 比较合适」

> 这是经验规则，不是绝对：

- **≤13B & GPU 显存≥80G（A100/H100）**
  - 优先：**ZeRO-2**
  - 如果你 batch 和 seq 都开很大，可以考虑 ZeRO-2 + gradient checkpointing
- **≈14B–30B & 单卡显存在 40–80G 左右**
  - 想全参调优：**ZeRO-3 +（可选）CPU offload**
  - 不想折腾：走 **QLoRA + LoRA**，ZeRO-2 甚至纯 DataParallel 也可以
- **≥30B 或者多卡大集群训练**
  - 全参：一般会用 ZeRO-3 / FSDP / Megatron-LM 混搭
  - 工程复杂度上升很多，这是大厂 / 实验室级别。



### 2.5 单卡场景下的DeepSpeed作用

在**只有一块 GPU** 的前提下，DeepSpeed 主要能帮你：

1. **用 ZeRO-3 + Offload，把模型「拆」到 GPU+CPU 上，塞进更大的模型**
   - 把一部分参数 / 优化器状态搬到 CPU（甚至 NVMe），GPU 只放一部分；
   - 结果就是：**单卡也能微调原本放不下的 20B / 30B 模型（尤其是全参微调）**。
2. **通过内存管理 + 梯度累积，让你用更长的序列 / 更大的有效 batch**
   - 比如：8B/14B 模型本来只能 1024 长度 + 小 batch；
   - 加上 ZeRO 的一些内存优化后，可以开到 2048/3072 的 seq length 或更大的有效 batch。
3. **用统一的 DeepSpeed 配置和启动方式，未来无缝扩展到多卡**
   - 你今天在单卡上用 `deepspeed + ds_config` 跑通；
   - 以后租到 4×H100，只要把 `--num_gpus 1 → 4`，ZeRO 管理好跨卡 sharding，你脚本几乎不用改。

**但要注意：**

- ZeRO-1/2 在单卡下基本只是「结构上开启了，但不会切 shard 到多卡」（因为没多卡可以分）——显存节省有限；
- 真正让单卡收益大的，是 **ZeRO-3 + CPU/NVMe offload** 这一套；
- 深度通信优化（1-bit Adam、跨机通信重叠）这些，在单卡是用不上的。

#### 2.5.1 单卡场景下 ZeRO 各个 Stage 实际效果

我们聚焦在「只有一块 H100 / A100」的情况：

**a)  ZeRO-1 / ZeRO-2**

- 设计初衷是「在多卡之间」把 **优化器状态** 和 **梯度** 切片：
  - ZeRO-1：shard optimizer states
  - ZeRO-2：再 shard gradients

**单卡时会发生什么？**

- 没有第二块卡可以分片，所以理论上的 shard 退化成「全在自己身上」；
- 一些实现细节可能仍然有小幅显存优化（比如内部 buffer 管理），但**效果非常有限**；
- 一般来说，**用 ZeRO-1/2 的意义主要在于：写好了以后，将来多卡训练直接复用。**

> 换句话说：**ZeRO-1/2 在单卡身上的核心价值是「未来扩展路径」，不是马上省一半显存。**

**b) ZeRO-3 + CPU Offload：单卡真正有用的一档**

ZeRO-3 本质是：**连模型参数也 shard 了**。在只有一块 GPU 的情况下，它不会把参数分到别的 GPU 上，而是可以配合：

- **offload_param 到 CPU**
- **offload_optimizer 到 CPU**

于是显存布局变成：

- GPU 只放：当前正在计算所需的一部分参数 + 部分激活；
- CPU 内存：放剩下那些暂时用不到的参数 / 优化器状态；
- 每次需要某层时，再从 CPU 拿回来（异步复制 + 计算 overlap）。

这时的收益就很明显了：

- 你可以在**单卡 40G / 80G** 上硬上一个本来光参数就 60–70G 的模型；
- 虽然训练会变慢（GPU⇄CPU 的数据搬运是瓶颈），但**至少能「跑起来」**。

所以：

> **如果你想在单卡 H100 上「勉强」全参微调 30B 模型，DeepSpeed（ZeRO-3 + offload）就是关键工具。**



#### 2.5.2 速度 vs 显存：单卡 DeepSpeed 的现实 trade-off

这点要说清楚，否则容易有误解。

**a) 在多卡时：DeepSpeed 通常是「又省显存又提吞吐」**

- 多卡之间分摊参数、梯度、optimizer；
- 通信和计算 overlap，整体吞吐显著提升；
- 可以把有效 batch 做得非常大，充分吃满 GPU。

**b) 在单卡时：DeepSpeed 大多是「牺牲一点速度，换更多内存空间」**

尤其是 ZeRO-3 + Offload：

- 优点：单卡能塞下比原本大很多的模型；
- 代价：
  - 每一层都要 GPU⇄CPU 拿参数；
  - 一顿 all-gather / scatter（虽然单卡不会有跨 GPU 通信，但会有 host 端调度）；
  - 前向/反向比「纯 GPU 内存训练」慢很多。

所以建议是：

- **如果你的模型本来就能轻松塞进 80G**（比如 Qwen3 7B/8B/14B）：
  - 不需要用特别激进的 ZeRO-3 + heavy offload；
  - 可以只用 **轻量的 DeepSpeed 配置 / 或干脆用 HF+accelerate**。
- **如果模型刚好塞不下**（比如你发现 14B / 32B 一上来就 OOM）：
  - 这时就有必要启用 ZeRO-3 + offload，把 CPU 内存当「外扩显存」。

<br>

## 3. vLLM 是什么？和 DeepSpeed 有啥本质区别？

**vLLM** 是一个专门为大模型推理（inference / serving）设计的引擎，核心目标是：

> **同样的模型，在同样的硬件上，用更少显存、跑出更高的吞吐（QPS / tokens per second）。**

它最核心的贡献是一个叫 **PagedAttention** 的机制：

### 3.1 为啥推理要单独一个「引擎」？

大模型推理和训练的瓶颈不一样：

- **训练瓶颈**：前后向计算、参数更新、梯度同步
- **推理瓶颈**：
  - 主要是 KV cache（各层注意力的 key/value）堆得特别多
  - 请求多时，显存容易碎片化（尤其是不同长度的请求混在一起）

普通 HF `generate()` 在大量并发请求时，很快遇到：

- KV cache 存不下、显存碎片严重
- batch 间 padding 严重，算力浪费
- 动态批次（request arrive anytime）调度困难

### 3.2 vLLM 的几个关键点

1. **PagedAttention：KV cache 的分页管理**
   - 把 KV cache 切分成「类似虚拟内存页」的块
   - 动态给不请求分配/回收这些块
   - 避免了显存碎片和过度 padding
   - 支持「动态批处理」（dynamic batching）：请求随时来，随时加入计算批次，而不会造成大量闲置 padding
2. **高吞吐 + 高并发**
   - 动态 batching + 分页 KV，用同样显存能支持更多并发请求和更高 QPS
   - 对 ChatGPT/LLM API 这种场景就是：“同样一张卡，服务更多用户”
3. **API 友好**
   - 提供 OpenAI-compatible API / Python API / CLI；
   - 支持直接从 HuggingFace 加载模型；
   - 和很多推理框架（如 FastAPI、Ray Serve）结合。

**一句话对比：**

- **DeepSpeed**：偏「训练/微调」侧（虽然也有 inference 模块），关注的是如何在训练时搞定参数、梯度、优化器。
- **vLLM**：偏「推理/服务」侧，关注的是如何在「生成 tokens」时管理 KV cache 和批次调度，把吞吐量拉满。

<br>

## 4. DeepSpeed & vLLM 可以混搭吗？怎么搭配最合理？

### 4.1 逻辑层面上：它们负责完全不同阶段

- **训练 / 微调阶段（SFT / RLHF）**：

  - 你用 **DeepSpeed+ZeRO** 把 7B/14B/32B 模型训好；
  - 得到一个 `pytorch_model.bin` or `safetensors` 的模型权重。

- **推理 / 部署阶段**：

  - 把这个训练好的模型 **转换为 vLLM 能加载的格式**（实际上大多数情况下就是标准 HF 格式）；

  - 启动 vLLM server：

    ```
    python -m vllm.entrypoints.openai.api_server \
      --model /path/to/your/finetuned-model \
      --port 8000
    ```

  - 然后你的应用侧用 OpenAI-style API 调用它。

所以，从工程流程看，最典型的组合是：

> **「DeepSpeed 负责训练 / 微调 → vLLM 负责推理 / 部署」**

这当然是 **可以混搭，而且非常推荐的组合**。



### 4.2 能不能在「一个程序里」同时用 DeepSpeed 和 vLLM？

一般来说：

- **不会这么干**，因为它们解决的问题不一样：
  - DeepSpeed 在做的是「训练期的分布式调度」；
  - vLLM 在做的是「推理期的 KV cache 管理和请求调度」。

如果你真的想搞：

- 在训练时用 DeepSpeed，训练完之后在同一个 Python 进程里再起一个 vLLM 推理引擎：
  - 技术上有可能，但非常不常见、没什么工程收益。
  - 通常是 **训练和推理分成两个独立服务**：
    - 训练集群（DeepSpeed on H100/A100 集群）；
    - 推理集群（vLLM + K8s / Ray / 单机）。



### 4.3 DeepSpeed 自己不是也有 inference 模块吗？为什么还要 vLLM？

是的，DeepSpeed 也有 **DeepSpeed-Inference**，可以进行推理加速（包括 Tensor-Parellel、KV cache 优化等）。但 vLLM 这套是「专注于 serving 的完整引擎」：

- vLLM 主打的是：
  - 动态批处理 + KV 分页
  - 高 QPS 在线服务
  - OpenAI-style API
- DeepSpeed-Inference：
  - 更适合大规模参数的模型验证 / 离线推理
  - 和它的训练栈整合更紧密（比如 Megatron-DeepSpeed 工作流）

对你这种「个人研究 + 以后可能做一个 demo / API」的场景，我会建议：

- **训练 / 微调：DeepSpeed（ZeRO-2/3）**
- **推理 / 接口服务：vLLM**



## 5. 实例总结

假设现在我们有：

- 有 8k 左右高质量数据；
- 想在 H100 上试 bf16 全参 SFT；
- 想从 8B 玩到 14B，顺便摸一下 32B；
- 未来很可能希望把模型挂成一个 API / demo。

建议的组合路线：

1. **训练侧（DeepSpeed）：**
   - Qwen3-8B / 14B：
     - 单卡 H100 80GB + **ZeRO-2** 起步；
     - 14B 可以试试 ZeRO-3 + offload 感受差异；
   - 32B：
     - 优先用 **QLoRA + LoRA**（轻量微调）；
     - 若真要全参，考虑多卡 ZeRO-3，这已经是「实验室级项目」了。
2. **推理侧（vLLM）：**
   - 把 DeepSpeed 训好的模型保存成 HF 格式；
   - 用 vLLM 起一个推理服务：
     - `--model` 指向你的 finetuned 模型目录；
     - 通过 OpenAI-style API 或 Python 客户端调用；
   - 同一张 H100，将来可以同时托管多个模型（例如一个 8B、一个 14B）。