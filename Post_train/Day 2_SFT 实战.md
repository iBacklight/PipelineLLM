# Day 2 SFT QLoRA微调实战

[TOC]



## 前置：概念速讲

**Assistant-only loss mask（仅助手侧损失）**
 SFT 本质是因果语言建模，但只对“助手角色（assistant）”产生的 token 计损失；对 user/system（以及 prompt 部分）不计损失。这样可以避免模型去“背诵”用户输入，强化“从指令到回复”的映射。实现上通常把非 assistant 段的 `labels` 置 `-100`（即忽略梯度）即可。我们的计划文档也明确了 Day2 的这个要点：SFT 的本质=因果 LM + assistant-only loss mask，并说明“不对 user/system 计损失（避免复述）”  。

**QLoRA（4-bit 量化 + LoRA 低秩适配）**
 核心是：把**基座模型权重**做 4-bit（nf4）量化并冻结，只训练插在某些线性层上的 **LoRA 低秩矩阵**（rank=r），再配合缩放系数（α）和 LoRA dropout 控制容量/正则。先量化，再lora微调。优点：显著降显存（单卡 16 GB 就能训 7B/8B），训练稳定、可拔插回退；我们的计划文档强调了“LoRA 低秩近似，r/α/Dropout 的作用；QLoRA 的 4-bit nf4 量化与误差补偿”  。

> 详见Intro/llm_processes/LoRA

英文版，推荐HF官方：https://huggingface.co/docs/trl/main/en/sft_trainer

---

## 0. 目标与产出

**目标**
	单卡(笔者使用的是Geforce RTX 4080)对 4B 指令基座（Qwen3-4B-Instruct-2507 ）做一次高质量 SFT，使用`TRL`标准训练库并采用 **assistant-only CE + QLoRA**，得到一个可拔插的 **LoRA 适配器**，并产出一张**基线评测表**（PPL、可用性、JSON 合法率）。 

------

## 1. 环境固定（Linux + CUDA 12.x）

```shell
# 使用conda 或者 venv 建立虚拟环境
conda create -n llm python=3.11 -y
conda activate llm

python -m pip install --upgrade "torch>=2.3" "transformers>=4.43" "datasets>=2.20" \
  "accelerate>=0.32" "trl>=0.9" "peft>=0.12" "bitsandbytes>=0.43" "evaluate>=0.4" "tqdm"
  
python -m pip install --upgrade unsloth unsloth_zoo # 使用unsloth
python -m pip install vllm # 部署和测试模型
```

------

## 2. 数据规范

### 2.1 数据格式（chat 格式，强制 role 明确）

#### 2.2.1 文件组织与基本格式

- **存储格式**：`train.jsonl` / `val.jsonl`（一行一条样本，UTF-8）。
- **字段**：最少包含
  - `id`：指定数据的合法编号
  - `messages`: `[{role, content, ...}, ...]`（严格区分 `system` / `user` / `assistant` / `tool`）
  - 可选元数据：`tags`、`domain`、`difficulty` 等，便于抽样与评测分桶。
- **生成文本**：训练时用各自模型的 **chat template** 将 `messages` 渲染为序列；[Qwen 官方文档](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)强调**必须使用指定模板**，否则行为有可能不可控。

#### 2.1.2 最小可用模板实例（注意：这是单轮指令的template）

```json
{"id":"sft-0001",
 "messages":[
   {"role":"system","content":"You are a helpful assistant."},
   {"role":"user","content":"简要解释什么是 QLoRA，并给出 1 个优点和 1 个注意点。"},
   {"role":"assistant","content":"QLoRA 将基座权重量化为4bit并冻结，只训练插入的LoRA低秩... 优点：显存占用低；注意：需正确选择rank与量化dtype。"}
 ]}
```

- 训练时使用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` 得到训练文本；

- **assistant-only mask**：只对最后一条 `assistant` 文本计损失；

  > 非 assistant 段 `labels=-100`，原因请见llm_processes/pre_train_and_sft.md文档，

- HF 的 [chat-templating 文档与社区](https://huggingface.co/docs/transformers/en/chat_templating)讨论展示了 Qwen 家族模板如何插入 `<|im_start|>...<|im_end|>` 这类 turn 标记，可用于定位响应边界。

#### 2.1.3 多轮对话模板（保留上下文，最后一轮计损失）

```json
{"id":"sft-0103",
 "messages":[
   {"role":"system","content":"You are a precise and concise assistant."},
   {"role":"user","content":"我有一份 CSV，列是 name, score，如何用 Python 读？"},
   {"role":"assistant","content":"可以使用 pandas: import pandas as pd; df = pd.read_csv('file.csv') ..."},
   {"role":"user","content":"再给一个不依赖 pandas 的纯 Python 版本。"},
   {"role":"assistant","content":"可以用内置 csv 模块：\nimport csv\nwith open('file.csv', newline='') as f:\n  reader = csv.DictReader(f)\n  rows = list(reader)\nprint(rows)"}
 ]}
```

> 多轮训练通常**只对最后一个 assistant 段计损失**（如果对每个回合的 assistant 都计损失，要确保模板能逐段正确打 mask）。Qwen3 的模板在“多个 assistant 消息”情况下发生过小问题的[修复讨论](https://huggingface.co/Qwen/Qwen3-1.7B/discussions/9)，我们的数据与评测也应覆盖这类样例以防模板边界错配。

#### 2.1.4 JSON 结构化输出（强约束样式）

```json
{"id":"json-0031",
 "messages":[
   {"role":"user","content":"仅输出一个JSON对象，含 name(string), score(float)。name=Alice, score=93.5。不得输出额外文字。"},
   {"role":"assistant","content":"{\"name\":\"Alice\",\"score\":93.5}"}
 ],
 "tags":["json-only","strict"]}
```

- 这类样本用于训练严格格式控制；验证集上统计 JSON 合法率。
- 另外可能需要添加负例来辅助验证结果。

#### 2.1.5 工具调用 / 函数调用

Qwen3 的模板支持**更稳定的工具参数序列化**。构造数据时，把“模型决定调用的函数”写在 `assistant` turn 的 `tool_calls` 字段，把工具返回写成一个或多个 `tool` turn。HF 的 Qwen-3 模板深度解读与 [PR/issue]( https://huggingface.co/blog/qwen-3-chat-template-deep-dive) 里都展示了思路。**示例（简化版）**：

```json
//tool schedma
{
  "id": "tool-0201",
  "tools": [ // tools calling from here
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather by city and day.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "day":  {"type": "string", "enum": ["today","tomorrow","yesterday"]}
          },
          "required": ["city","day"], //possibly "loc"
          "additionalProperties": false
        }
      }
    }
  ],
  "messages": [
    {"role": "system",  "content": "You can call tools to get weather."},
    {"role": "user",    "content": "明天 Edmonton 天气？"},

    {"role": "assistant", "content": "", "tool_calls": [ // call tools
      {
        "type": "function", 
        "id": "call_1",
        "function": {"name": "get_weather", "arguments": {"city": "Edmonton", "day": "tomorrow"}}
      }
    ]},

    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\":12,\"condition\":\"Cloudy\"}"},

    {"role": "assistant", "content": "Edmonton 明天多云，约 12°C。"}
  ]
}

```

- **计损失**：一般只对最后一条自然语言回复的 `assistant` 段计损失；工具调用时的“函数签名“和”参数”的生成可选是否计损失。
- **一致性**：工具样本要成对出现（assistant 触发 → tool 响应 → assistant 归纳）。

#### 2.1.6 “思考和非思考”模式（Qwen3 的新特性）

Qwen3 在 `apply_chat_template` 时支持 `enable_thinking=True/False`（是否插入“思考 token/段”）；模型卡与讨论显示了这个开关的用法。**数据集层面**我们可以准备两类样本：

- **非思考**：普通指令对话（默认就好）；
- **思考样本**：把“推理草稿/链路”放进 assistant 文本的“思考区域”，再给出可见答案（更像 CoT）。使用`reasoning_content` 编辑思考和思维链的内容
   训练或推理时切换 `enable_thinking`。

**思考样本（简化）**

```json
{
  "messages": [
    {"role": "user", "content": "20 个苹果，送出 7 个，还剩多少？"},
    {
      "role": "assistant",
      "content": "13",
      "reasoning_content": "共有 20 个苹果，送走 7 个，剩下 20 - 7 = 13。"
    }
  ]
}
```

> 训练时是否真的需要“思考段”由我们决定——Qwen3 的模板**不强制**思考。[Hugging Face](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)

### 2.2 Qwen3 的特殊控制符（special tokens）以及token生成

#### 2.2.1 Qwen3 的特殊字符 / special tokens

在 HuggingFace 上加载 **Qwen3 tokenizer** 后，可以用下面方式查看它定义的特殊符号：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    torch_dtype="bfloat16",   # 或者 "auto"
    device_map="auto",
    trust_remote_code=True
)

print(tok.special_tokens_map)       # 显示 bos/eos/pad 等特殊符号
print(tok.additional_special_tokens)  # 显示像 <|im_start|> <|im_end|> 之类的控制符
```

典型输出会包含：

- `<|im_start|>`、`<|im_end|>`：标记一段 message 的开头和结束
- `<|system|>`、`<|user|>`、`<|assistant|>`：角色标签（有些版本直接作为 role 字段被渲染）
- `<|endoftext|>`：等价于 `<eos>`
- 可能还有 `<|extra_0|>` … `<|extra_99|>` 预留符号

Qwen3 的 **chat 模板** 就是用这些特殊符号把 `messages=[{role, content}]` 渲染成连续的文本序列。

#### 2.2.2 数据集里要不要直接写这些符号？

**不要**。我们在构造数据集时，应该保持干净的 JSON 格式：

```json
{ // 有的时候role可以放在后面
  "messages": [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":"今天天气如何？"},
    {"role":"assistant","content":"今天天气晴朗，气温25℃左右。"}
  ]
}
```

然后在训练时调用：

```python
tokenizer.apply_chat_template(
    example["messages"], 
    tokenize=False, 
    add_generation_prompt=False
) # 自动插入特殊符号
# 注意，一般情况下，<think><\think>不算qwen3的规定特殊字符，需要自行添加
```

Tokenizer 会**自动插入** `<|im_start|>`、`<|im_end|>` 这些控制符。这样做有两个好处：

1. **模板稳定**：不同版本的 Qwen 可能更新模板，如果我们手工写符号，将来不兼容。
2. **避免错位**：chat 模板会自动保证 assistant/user 的边界正确，利于 loss mask（只给 assistant 段打标签）。

#### 2.2.3 Assistant-only loss mask 和特殊符号的关系

当 `apply_chat_template` 生成序列后，里面会包含 `<|im_start|>assistant ... <|im_end|>`。

- collator 会用这些锚点来确定从哪里开始打 `labels`，无需自己手动再训练时标记（数据中当然要有labels）。
- `<|im_start|>` 本身通常 **不计损失**（它属于模板 token，不是答案）。
- **只对 assistant 段正文**打损失。（当我们使用TRL SFTConfig的时候， 只需要标记`"assistant_only_loss": True` ，无需自行应用）

#### 2.2.4  如果我们想自己确认模板

可以直接打印一条样本看看：

```python
sample = [
    {"role":"user","content":"请给我一个Python打印Hello的例子"},
    {"role":"assistant","content":"print('Hello')"}
]
print(tok.apply_chat_template(sample, tokenize=False))
```

输出大概类似：

```shell
<|im_start|><user>
请给我一个Python打印Hello的例子<|im_end|>
<|im_start|><assistant>
print('Hello')<|im_end|>
# 模型只会计算assistant之后的CE
```

> 所以 `<|im_start|>` `<|im_end|>` 这类特殊符号只会在 **最终训练文本里**出现，由 tokenizer 模板自动插入；我们自己准备的数据集 JSON 不需要也不应该直接写它们。

------

## 3. 模型，数据集与 tokenizer

### 3.1 模型

我们使用**unsloth/Qwen3-4B-Instruct-2507**，其中`Instruct` 指的是 **instruction-tuned（指令微调版）**。具体来说，它在 Base 基础上，用人工或合成的**指令–回答数据集**训练过，所以我们在做微调的时候，不需要从0开始教模型怎么回答。这是一个dense模型，没有reasoning（thinking）能力，但是根据官网的描述：

> **Significant improvements** in general capabilities, including **instruction following, logical reasoning, text comprehension, mathematics, science, coding and tool usage**.

该4b模型在多项领域均有提升，且在多项任务中击败了gpt 4.1 nano和Qwen3-30B-A3B Non-Thinking，训练需要的资源很少，非常适合我们上手做小规模的测试和QLoRA微调。

- **Qwen 系列**常见响应模板锚点（用作 mask 的 response 标记）近似为：`<|im_start|>assistant` 后的正文；

可以使用`transformers`(huggingface)下载模型（注意：因为我们需要unsloth来加速推理，注意选择模型的前缀为unsloth，而不是qwen官方发布的4B模型）

```python
"""
you can clean your space first using:
conda clean --all
pip cache purge
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST = False

model_name = "unsloth/Qwen3-4B-Instruct-2507"
save_dir = "path/to/models"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=save_dir
)
```

接下来可以使用`vllm`进行部署。vLLM 是一个专门用来 **高效推理大语言模型（LLM）的推理引擎**。它不是训练框架，而是针对“部署和服务端推理”优化的，解决的是 Hugging Face `transformers` 在长上下文、多并发场景下速度慢、显存占用高的问题。

> 关于这一部分，可以详见Background/Training_infra文档。

```shell
# 在4080卡下，以下设置会占用：
#|=========================================+======================+======================|
#|   0  NVIDIA GeForce RTX 4080        Off | 00000000:01:00.0  On |                  N/A |
#|  0%   39C    P8              14W / 320W |  14439MiB / 16376MiB |      1%      Default |
#|                                         |                      |                  N/A |
#+-----------------------------------------+----------------------+----------------------+
# 注意，这里只是最大占用，实际进行训练和推理不需要这么大
# note this path is to the directoty that can directly see config.json
vllm serve /path/to/models/snapshot \ 
  --max-model-len 16384 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --enforce-eager
# 若版本或者硬件支持，再加上这一条（KV 内存约减半）：
#  --kv-cache-dtype fp8下载部署完成后，可按照deploy.py中的vLLM test流程进行openai api的测试，以了解模型情况。同时可以时所用[evalscope](https://github.com/modelscope/evalscope)  来进行压力和性能测试，详见eval。
```

### 3.2 数据集

#### 3.2.1 数据集类型

为了使得模型微调后，依然具有混合推理能力，需要考虑在数据集中加入普通对话和带有推理字段的数据集（可以拼接）。由于4b-instruct不是推理模型，所以不需要加入推理数据集。我们使用[shareAI/DPO-zh-en-emoji](https://hf-mirror.com/datasets/shareAI/DPO-zh-en-emoji)数据集，用来加如更多的emoji，再添加两个数学数据集合cmath/MAWPS。以下是其他选项。

| 名称         | 语言 / 特性                       | 是否带思考链 / 推理标注                                      | 适合用作对话微调吗                                    | 备注 / 可取部分                                              |
| ------------ | --------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| **CORECODE** | 中文，共识推理 + 对话             | 有共识知识注释（对话里包含推理判断）([arXiv](https://arxiv.org/abs/2312.12853)) | 是，可以把其对话 + 推理判断抽出来做 mixed 模型训练。  | 这个数据集偏“对话中涉及常识判断” rather than 明显的自我思考链，但足以作为推理能力的一部分训练素材。 |
| **MuTual**   | 中文 / 英文混合，多轮对话推理选题 | 它是多轮对话推理选择题，要求模型判断哪一个选项最合理 ([GitHub](https://github.com/Nealcly/MuTual)) | 较适合做**理解 / 判断模块**训练，而非直接对话回复模型 | 可以把问题 + 选项转成对话 prompt 格式，模型回复“正确选项 + 推理链”。 |
| **KdConv**   | 中文，多领域知识驱动对话          | 主要是知识对话，不一定带显式思考链                           | 是，对话能力提升 + 知识互动增强                       | 可作为对话基础语料（不强取推理链）([GitHub](https://github.com/YouTaoBaBa/Chinese-Dialogue-Dataset)) |
| **LCCC**     | 中文对话 / 社交闲聊               | 基本对话，没有思考链标注                                     | 是，对话能力训练基石                                  | 可作为“通用对话”数据的覆盖层，保证模型在对话流畅性方面不过拟合推理行为([ResearchGate](https://www.researchgate.net/publication/346107804_A_Large-Scale_Chinese_Short-Text_Conversation_Dataset)) |

#### 3.2.2 数据集处理与合并

为了更加偏向工业界实际的操作方法, 我们这样构建多个数据集：

- **先“规范化→分库存盘”**：把每个原始数据集清洗成统一 schema（`messages`, `source`, `tags`…）后，分别 `save_to_disk`。这样：
   - 复用便捷：后续可以用不同权重、多套配比**快速重混**，不用反复解析 JSON/CSV。
  - 训练高效：`load_from_disk` 直接读 Arrow，性能更好，也易于多人协作与版本管理。
- **按需“构建混合包”**：针对某次训练，把分库存盘的数据用**权重**混起来，再 `save_to_disk`（或导出 JSONL 给外部框架）。
  - 若只想“按比例大致抽样”，用 **interleave**。
  - 若要“严格命中配比”，用 **resample**（带放回采样，精准满足权重）。

一次性把所有数据“合并成一个数据集”固然也能训，但不利于后续快速换权重和复现实验。接下来，我们一步步实践如何对数据作处理。

**1) 规范化（normalize_*，非归一化）**

把三个原始数据集转成统一 schema：每条样本都是

```json
{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
```

同时清洗文本（去控制符、截断超长答案等）。这样可以

- 统一成对话式 `messages`，直接喂到绝大多数 SFT/指令微调 Trainer。
- 避免不同源的字段名不一致（如 `question` vs `Question`）。
- 便于后续可视化、对齐 tokenizer、统计长度等。

**例子：**

- CMATH：`{"question":"芳芳有99页…","golden":"9"}` → `messages=[{user:"…?"},{assistant:"9"}]`。
- MAWPS：把 `Question` 里的 `N_0, N_1` 用 `Numbers` 替换成具体数值，再生成 `messages`。
- Emoji：优先用 `answer_zh`，没有就回退 `answer_en`；过长答案截断到 3000 字符，防止极端样本拖垮 batch。

**2) 加标签（add_metadata）**

为每条样本增加：

```python
source="cmath"/"emoji"/"mawps"
tags=["math","zh"] 等
```

- 训练/评测分桶：看“math/zh/en/chat”等子域效果。
- 后续重混时按来源设权重更清晰。
- 排障与复现（看到一条异常样本，知道它来自哪一池）。

**例子：**

- CMATH → `source="cmath", tags=["math","zh"]`
- MAWPS → `source="mawps", tags=["math","en"]`

**3) 去重（deduplication）**

对 `messages` 做 MD5，按哈希去重（先各自去重，再合并后再去重一次）。

```python
# 计算字符串 s 的 MD5 哈希值，并将其封装在一个字典中返回。
# MD5 是一种广泛使用的加密散列函数。无论输入 s 有多长，它都会产生一个固定长度（128位）的散列值。
return {"_h": hashlib.md5(s.encode("utf-8")).hexdigest()}
# Python 的 hashlib 库（以及大多数哈希算法）只能处理字节数据，不能直接处理文本字符串。如果直接传字符串会报错。所以这里用encode将字符串变量 s 转换为 字节流 (bytes)。
```

- 防止某些数据集内部或跨数据集重复（尤其是爬取/复刻的公开数据）。
- 重复样本会导致过拟合、指标膨胀。

**例子：**

- 如果 CMATH 和 MAWPS 恰好都有一道“99页看了90页”的题，只留一条。

**4) 分库存盘（可选，separate_save）**

把规范化+去重后的每个子集 `save_to_disk` 为 Apache Arrow。这种格式：

- 它把数据存成一种 **列式表格**（类似 Pandas DataFrame），而不是一行行的 JSON/CSV。
- 这种格式在内存里很紧凑，适合**高效读取和随机访问**，尤其是大规模数据。
- HuggingFace 的 `datasets` 库底层就是用 Arrow 存数据，所以 `save_to_disk` 输出的其实就是 Arrow 格式的目录。

**为什么：**

- 工业界常见做法：**规范化数据集作为“池”长期保存**。
- 后续任意训练都从这些池按权重重混，无需反复清洗原始 JSON/CSV。
- Arrow 读写快、内存友好、团队可复用。

**例子：**

- 输出：`datasets/normalized/cmath/`, `…/emoji/`, `…/mawps/`
- 下次想提高英文数学权重，只需 `load_from_disk` 回来重混即可。

5）**混合（interleave / resample）**

有两条路：

>**5.1) 交错混合 `interleave_mix`**
>
>- 先把各池 `shuffle(seed)`。
>
>- 按 **概率**（由权重归一化而来）交替抽样，直到“全部耗尽”。
>
>**为什么：**
>
>- 快速、稳定，不会对小池做大量放回上采样。
>- 训练时整体配比“近似命中”，适合日常迭代。
>
>**例子：**
>
>- 权重 `cmath:0.4, emoji:0.3, mawps:0.3` → 近似 4:3:3 的交错顺序，如
>  `cmath, emoji, cmath, mawps, cmath, emoji, …`
>- 如果某个池很小，耗尽后就不再抽它，剩下的按相对概率继续。
>
>**5.2) 严格配比 `resample_mix`**
>
>- 设定 `total_size`（不设则默认按总量的 0.9）。
>- 按权重计算目标样本数，对各池**放回采样**到目标数，最后 `concatenate`。
>
>**为什么：**
>
>- 精准命中配比，便于可复现实验与 A/B。
>- 代价：小池会被放大，有过拟合风险。
>
>**例子：**
>
>- 总量 30k；权重 0.4/0.3/0.3 → 采样 12k/9k/9k 条；
>- 若 `emoji` 只有 2k 原样本，也会“放回”抽到 9k，需注意过拟合，通常配合强正则/更高温度/多轮 shuffle。
>

**6) 终轮去重（merged 后再次 dedup）**

混合完成后，再跑一遍哈希去重。

**为什么：**

- 防止由于不同池里残留的重复在合并后暴露。
- 也避免某些边界情况下 interleave/resample 产生意外重复。

**例子：**

- 两个池各留了一份同题样本，合并后删掉一份，确保训练集唯一性。

**7) 切分（split_train_valid）**

对合并数据 `shuffle(seed)`，然后按比例切出验证集（下限 100 条，避免 val 过小）。

- 统一的验证集用于早停/对比。
- 下限保护：即使总量不大也能得到有意义的验证统计。

**例子：**

- 总 50k，`valid_ratio=0.02` → 1k 验证、49k 训练。
- 若总量 3k，仍会给到至少 100 条验证。

**8) 随机性与复现（seed）**

- `shuffle(seed)`、放回采样时都用确定性随机源。

**为什么：**

- 复现实验。
- 不同 run 想“多样化”也可以改 seed。

**例子：**

- A/B 两组只改 `mix_weights`，保持 seed 相同，便于对比；
- 想生成“第二版训练包”，就把 seed +1。

**9) packing**

对于极端的文本数据，会占用大量的显存空间，导致显存利用效率低下。这时候我们可以使用packing将短的数据合并，从而节约空间。但是需要注意的是，对应的mask也要改变：
![packing](pics/packing.png)

Fig.1 Packing. 该图片取自于余老师的github repo: [LLM-RL-Vis](https://github.com/changyeyu/LLM-RL-Visualized)。我们推荐您浏览老师的github或者购买这本书。

**10) 导出（save_to_disk / to_json）**

**做了什么：**

- 把最终 `train/validation` 用 Arrow `save_to_disk`；
- 也可 `to_json(force_ascii=False)` 导出 JSONL。

**为什么：**

- Arrow：加载快、少内存、HuggingFace 生态友好；
- JSONL：兼容非 HF 的 Trainer/Pipeline。

**例子：**

- 产物目录：

  ```bash
  datasets/_qwen3_sft_mixed/train/    # Arrow
  datasets/_qwen3_sft_mixed/validation/
  datasets/_qwen3_sft_mixed/train.jsonl
  datasets/_qwen3_sft_mixed/val.jsonl
  ```

在这里，我们也可以使用pandas对数据进行处理，然后再使用pd.concat拼接数据，再用datasets库进行抽样。具体做法，请见`build_datasets.py`。

同时也可以使用`check_dataset.py` 来验证生成的数据集是否符合qwen3训练的规范。当然，HF也提供了官方的数据处理方法，如果我们不想自己处理，也可以使用**Hugging Face Datasets**：

-  `datasets` 包本身就支持很多内置操作：
  - `.filter()` → 清理异常样本
  - `.map()` → 规范化字段
  - `.remove_columns()` / `.rename_column()`
  - `.drop_duplicates()`
- 特点：和 HF 生态无缝，数据会存成 Arrow，处理速度很快。

👉 用法例子：

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data.jsonl")["train"]
# 去重
ds = ds.drop_duplicates("text")
# 清理空样本
ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
```

------

## 4. 关键训练策略（对单卡4080显存友好，其他酌情调整）

- **量化**：从`from_pretrained`加载tokenizer和模型的时候，可以传入以下参数`load_in_4bit=True, quant_type="nf4", compute_dtype=bfloat16`（QLoRA），其中，quant_type="nf4"是一种 4bit 的分布感知量化（NormalFloat4），比普通 int4 保真度更好。

  > 但是，qwen3不支持该参数。所以对于我们的训练来说要去掉这一项
  >
  > ```python
  > MODEL_CONFIG = {
  >     "max_seq_length": 2048,
  >     "dtype": "bfloat16",
  >     "load_in_4bit": True,
  >     # "quant_type": "nf4",  # Removed - not supported by all models
  > }
  > ```

- **LoRA**：`r=16, alpha=32, dropout=0.05`（先小后大）。其中，

  - r 是LoRA 矩阵的秩，决定可训练参数量。较小（如 4–8）：显存更省，但拟合能力有限；较大（如 32+）：可能学得更好，但显存和训练时间增加。

  - alpha缩放系数，和 r 搭配。一般设成 `2*r`（这个数字实际上是一种经验调参）。值太低：训练不足；值太高：可能过拟合。

  - 将 `target_modules` 设置为覆盖**所有的线性层**（即Attention 部分 + MLP 部分）。我们的 QLoRA 使用了 4-bit 量化，虽然节省了显存，但多少会损失一点精度。微调更多的模块有助于模型更好地适应新数据，弥补基座模型的精度损失。

     ```python
     "target_modules": [
     "q_proj", "k_proj", "v_proj", "o_proj", # Attention layers
     "up_proj", "down_proj", "gate_proj" # MLP layers
     ],
     ```

- **序列**：`max_seq_len=1024`（或 2048，先从 1k 起步）每条样本最多 1024 token。显存线性增长。

- **显存三件套**：`packing=True` + `bf16=True` + `gradient_checkpointing=True`

  - **`packing=True`**→ 把多个短样本打包进一个序列，减少 padding，显存更省。
  - **`bf16=True`** → 计算用 bfloat16，稳定且省显存。
  - **`gradient_checkpointing=True`** → 在反向传播时节省中间激活，显存大幅下降，但计算时间增加 ~20–30%。可选
  
- **批量**：`per_device_train_batch_size=1` + `grad_accum=16` 起步（等效大 batch）。

- **优化器**：AdamW，`lr=1e-4 ~ 2e-4, wd=0.01, warmup_ratio=0.03`。

- **评测**：每 N 步在 `val` 上统计 **PPL + 可用性 + JSON 合法率**。

我们将所有参数都储存在`config.py`中，以更直观地观察参数之间的情况。

------

## 5. 训练脚本骨架（QLoRA + assistant-only + packing）

> BitsAndBytes 4-bit 加载；2) PEFT LoRA；3)**DataCollatorForCompletionOnlyLM** 通过**响应模板**做 assistant-only mask；4) `packing=True`。

接下来，我们开始正式进行QLoRA微调。我们的流程如下

```python
准备数据集(JSONL/Arrow) # 前面已经讲解过
    ↓
load_datasets / load_from_disk # 前面已经涉猎过
    ↓
数据规范化(messages 格式) # 前面已经涉猎过
    ↓
FastLanguageModel.from_pretrained(量化加载基座) 
    ↓
FastLanguageModel.get_peft_model(QLoRA 配置) 
    ↓
Trainer / SFTTrainer 定义(optimizer, lr, batch_size) 
    ↓
trainer.train() 进行 SFT 微调 
    ↓
保存 LoRA adapter (merge_and_unload 可选) 
    ↓
推理/评估/部署
```

首先测试unsloth对模型和tokenizer的加载情况，[Qwen3](https://qwen.ai/research)普遍使用 **BPE tokenizer**（Byte Pair Encoding）。

```python
model_args = {
                "max_seq_length": self.model_config.get("max_seq_length", 2048),
                "dtype": self.model_config.get("dtype"),
                "load_in_4bit": self.model_config.get("load_in_4bit", True),
                "trust_remote_code": True,
                "device_map": "auto"
            }
            
# FastLanguageModel.from_pretrained returns (model, tokenizer) tuple
self.model, self.tokenizer = FastLanguageModel.from_pretrained(
    self.model_path,
    **model_args
)

# print the model
print(model)

```

我们会看到诸如：

```bash
(Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 2560, padding_idx=151654)
    (layers): ModuleList(
      (0-35): 36 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        ...
        等显示模型结构的信息。
```

这里看到，模型已经被以4bit量化的形式加载下来了（原来是torch.bfloat16精度）。接下来，先搞清楚peft的概念：**PEFT = Parameter-Efficient Fine-Tuning（参数高效微调）**

- 它是一类方法的统称，不是单一算法。
- 目标：在不更新大模型全部参数的前提下，只训练很少的一部分参数（增量参数/适配器），从而高效地完成下游任务的微调。

在unsloth中：

- **from_pretrained**
  
  - 负责：下载/本地加载权重、构造模型与 tokenizer、设定 dtype & device_map、（可选）k-bit 量化、速度优化开关。
  - 输出：一个**可推理**的基础模型；还没“变成 LoRA 模型”。
- **get_peft_model**
  
  - 负责：根据 `peft_config` 给模型**插入可训练的适配层**（LoRA 等），并把**仅适配层**标记为可训练；必要时做 k-bit 训练准备（如 norm cast、use_cache=False）。
  - 输出：一个**可微调**的 PEFT 模型；优化器只会看到适配器的参数（显存/算力更省）。
  
  明白以后，我们把加载的与训练模型，转换成peft model，其实就是**设置`LoRA`的配置**。注意：`QLoRA` 的关键就是**加载时 4bit** + **挂 LoRA 适配器并只训它们**。在 Unsloth 里基本就是：`from_pretrained(load_in_4bit=True, quant_type="nf4", compute_dtype=bfloat16)` 然后 `get_peft_model(LoraConfig(...))`，无需额外手动步骤（Unsloth内部已做 k-bit 训练准备）。

```python
 # For Unsloth, we need to pass LoRA parameters directly
 self.model = FastLanguageModel.get_peft_model(
    self.model, 
    r=self.lora_config["r"],                    # LoRA rank
    target_modules=self.lora_config["target_modules"],  # Target modules
    lora_alpha=self.lora_config["lora_alpha"],  # LoRA alpha
    lora_dropout=self.lora_config["lora_dropout"],  # LoRA dropout
    bias=self.lora_config["bias"],              # Bias setting
    use_gradient_checkpointing="unsloth",       # Use Unsloth's checkpointing
    random_state=3407,                          # Random seed
    use_rslora=False,                           # Don't use rank-stabilized LoRA
    loftq_config=None,                          # No LoftQ
 )

 # Enable training mode
 self.model = FastLanguageModel.for_training(self.model)
```

接下来正式接入SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
```

在trainer的config设置方面，可以用到SFTConfig或TrainingArguments两种，我们这里推荐使用SFTConfig，原因详见Appendix第一个问题。

Recall: 之前提到过，SFT在训练时，为了防止重复用户的问题，通常在指令中做**loss mask**，即在loss计算的时候，只将assistance后面的内容记入CE loss, 而不计算来自user和system的内容（区别问答训练，这个会计算user的内容）。

```python
sft_args = {
    **modified_training_config,
    **self.sft_config,
    "output_dir": self.output_config["output_dir"],
    "run_name": self.output_config["run_name"],
    "logging_dir": self.output_config["logging_dir"],
}

# Log assistant loss masking status
if self.trainer_config.get("assistant_only_loss", True):# 只需在config.py中加入assistant_only_loss = True即可
    self.logger.info("Assistant loss masking ENABLED - loss will be computed only on assistant responses")
else:
    self.logger.info("Assistant loss masking DISABLED - loss will be computed on entire sequence")

training_args = SFTConfig(**sft_args)

# Initialize trainer with SFTConfig
self.trainer = SFTTrainer(
    model=self.model,
    tokenizer=self.tokenizer,
    train_dataset=self.train_dataset,
    eval_dataset=self.val_dataset,
    args=training_args,
)
```

最后进行训练

```python
# Start training
train_result = self.trainer.train()
```

------

## 6. 评测

### 6.1 主观评测

我们做了一个小的测试脚本`subj_test_model.py`，用来对QLoRA微调后的qwen3 4b模型进行主观测试。测试包括：

```python
🔧 What would you like to do?
1. Run predefined tests
2. Interactive chat
3. Both
4. Exit
Enter your choice (1-4): 
```
在一些预先设定好的问题后，可以看到模型对于数学问题比较冷漠，会直接回答答案，比较简洁

```
🤔 Input: Mary has 5 cats. She gives 2 cats to her friend. How many cats does she have left?
🤖 Output: 
3
```

在一些常见的问答环节中，qwen模型尝试加入更多的emoji表情，语气更加生动

```
🤔 Input: 什么是深度学习？请用简单的话解释。
🤖 Output:

😊

深度学习（Deep Learning）就是让机器学会像人一样思考和学习！ 🤖

想象一下，你是一个小孩子，正在学画画。我们的老师给我们一些画笔、颜料和纸张，然后我们开始画一个苹果🍎。

现在，假设我们有一个机器人朋友，它也想学画画。但是，它不知道怎么开始。于是，我们的机器人朋友会观察我们画画的过程，记住我们的动作和步骤，然后模仿我们画苹果的样子。

这就是深度学习的原理！ 💡 它使用很多层（layer）来处理数据，每一层都像一个老师，教机器人如何识别和理解信息。最后，机器人可以自己创作出新的作品！

所以，深度学习就像是一个超级聪明的机器人，能够从数据中学习和成长！ 🚀
```

在互动对话测试中，我们要求他自我介绍，相比于之前的回答

```
你们好呀！✨ 我是Qwen，是阿里云研发的超大规模语言模型。我们可以叫我通义千问，或者直接叫Qwen。我就像一个知识渊博的助手，能帮我们解答各种问题，无论是写故事、写公文、写邮件、写剧本，还是做数学题、编程、逻辑推理，甚至可以陪我们聊天、玩游戏，我都能尽力提供帮助。

我支持多种语言，包括中文、英文、德语、法语、西班牙语、葡萄牙语、俄语、阿拉伯语、日语、韩语等，可以和我们用我们最熟悉的语言交流。

无论我们是学生、老师、程序员、创业者，还是只是想找个聊天伙伴，我都很乐意陪我们一起探索世界、解决问题、激发灵感！

所以，有什么我可以帮我们的吗？😊  
比如：  
- 写一篇有趣的短文？  
- 帮我们写一封邮件或简历？  
- 解释一个复杂的概念？  
- 或者只是想聊聊天、放松一下？

随时告诉我我们的想法吧！🌟
```

现在的模型

```
我靠！😊 你好呀！我是通义千问，中文智者！ 🤖️ 我是一个超智能的语言模型，能够回答各种问题、提供帮助和陪伴我们聊天。 💬️ 我可以写诗、写小说、写文章、写代码等等！ 📝️ 我还能帮我们做数学题、科学实验、编程设计等等！ 🔍️ 所以，叫我通义千问吧！ 👋️
```

### 6.2 客观评测

我们需将保存的lora adapter和原base model合并，再调用vllm和evalscope进行测试。可以对模型的**各项指标**，以及**PPL (PPL 越低，说明模型在验证集上的语言建模能力越好)/ 回答可用性 / JSON 合法率** 进行客观测试。详见`obj_test_model.py`。

```shell
============================================================
🔍 OBJECTIVE MODEL EVALUATION
============================================================
📊 Evaluating model: /home/awpc/studies/models/unsloth/Qwen3/FTTrained
📁 Validation data: dataset/processed_datasets/qwen3_sft_mixed/val.jsonl

Computing PPL on dataset/processed_datasets/qwen3_sft_mixed/val.jsonl...
Generating train split: 100 examples [00:00, 16331.69 examples/s]
Processing sample 0/100
Processing sample 50/100
PPL computed on 100 samples
✅ Perplexity: 3.115
Computing usability rate on dataset/processed_datasets/qwen3_sft_mixed/val.jsonl...
Processing sample 0/100
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Processing sample 20/100
Processing sample 40/100
Processing sample 60/100
Processing sample 80/100
Usability rate: 1.000 (100/100)
✅ Usability Rate: 1.000

============================================================
📋 EVALUATION SUMMARY
============================================================
Perplexity: 3.115
Usability Rate: 1.000
```

------

## 7. 推理与合并

- **挂载式推理**：`base + adapter`（上面评测脚本即为此模式）。
- **权重合并导出**（可选）：将 LoRA 合并到基座，再导出全量 fp16/safetensors 便于部署；或维持适配器形态，节省磁盘与提升回退灵活性（也符合 Day 5 的“参数隔离/多适配器”路线）。

------

## 8. 显存预算与吞吐建议（4080 策略）

- **模型显存**：7B/8B + 4-bit 量化约 6–8 GB；加上激活/优化器/packing，一般 **12–15 GB**。
- **OOM 处理顺序**： 如果你的显卡在训练时，cuda爆出了Out of Memory (OOM) 的错误，根据优先级，我们依次对下述参数进行调整直到可以正常运行训练。
  1. `max_seq_len: 1024
  2. `grad_accum: 8 → 16`（等效大 batch）
  3. 降 `lora_r: 16`（和 `alpha`）
  4. 减少 eval 频率或 batch = 1


------

## 9. 训练监控与复现性

- **随机种子**：统一设置 `seed=42`（训练、数据打散、cuDNN），便于 Day 5/6 对照。
- **固定小评测集**：Day 2 开始就固定，后续 DPO（Day 3）、PPO/GRPO（Day 4）与 CL（Day 5/6）都复用做回归检测。 



## Appendix

### 1. SFTConfig vs. TrainingArguments

**`TrainingArguments` 本身不提供 “assistant-only loss” 开关**；这个开关在 **`SFTConfig`/`SFTTrainer`** 里。

官方给出的做法是：用会话数据 + `assistant_only_loss=True`（在 `SFTConfig` 中）就只对助手段落计损失。



### 2. 显存

**算力在吃满，但模型和数据放得“太省”了**，导致显存没有被充分利用。按照下表进行debug。

| 调整方向       | 参数 / 操作                                                  | 预期效果                    | 风险 / 注意事项                   |
| -------------- | ------------------------------------------------------------ | --------------------------- | --------------------------------- |
| **批量大小**   | `per_device_train_batch_size: 2 → 4 → 8`                     | 显存线性增加，吞吐更高      | 可能 OOM，可配合减少 `grad_accum` |
| **梯度累积**   | `gradient_accumulation_steps: 8 → 4 → 2`                     | 单步显存↑，等效 batch 不变  | 训练曲线更抖动，需观察收敛        |
| **序列长度**   | `max_seq_length: 2048 → 4096`                                | 显存和计算量 ~2×            | 数据平均长度必须够长，否则浪费    |
| **packing**    | `packing=False → True`                                       | 显存利用率↑（减少 padding） | 要确保 tokenizer 有 `eos_token`   |
| **LoRA 容量**  | `r=16 → 32 → 64`，`target_modules` 增加 `up_proj/down_proj/gate_proj` | 可训练参数↑，显存↑          | 过大可能过拟合 / 训练变慢         |
| **量化等级**   | `load_in_4bit=True → load_in_8bit=True` → 全精度 (bf16)      | 显存消耗翻倍，训练更稳定    | 显存可能不足                      |
| **优化器实现** | `adamw_8bit → adamw_torch`                                   | 显存略↑                     | 内存开销更大，速度可能慢          |
| **长序列填充** | 检查数据平均长度，必要时扩展数据                             | 利用率更高                  | 如果数据短小，靠扩长无意义        |

推荐优先顺序

1. **增大 batch size**（最稳）。
2. **检查数据长度分布 + 开启 packing** → 确保序列真正利用。
3. 如果显存仍然很空：**提高 LoRA r 或 seq length**。
4. 实在不怕显存爆，就**关掉 4bit，换 8bit / bf16 全参**。



### 3. 【重点必看】通过多种参数来确定实际的 Batch Size （实例论证）

Batch size直接与两个参数相关

1. **`per_device_train_batch_size`**

- **含义**：每张 GPU 在一次前向 + 反向传播中能处理多少条样本。
- **直观理解**：就是“显卡单步能塞多少样本进去”。
- **受限因素**：主要是 **显存大小** 和 **样本长度 (`max_seq_length`)**。

例子：

- 如果我们使用单卡训练，开 `per_device_train_batch_size=4`，意味着每次梯度计算就处理 4 条样本。（16GB以下可能会OOM）
- 如果显存爆了，就要调小，比如设成 1 或 2。

2. **`gradient_accumulation_steps`**

- **含义**：累积多少步的梯度，再进行一次参数更新。注意，每一次累积的时候，梯度值还是计算的，只不过不做更新。
- **直观理解**：显存不够时，用小 batch 反复跑几次，把梯度加起来，相当于模拟大 batch。

例子：

- `per_device_train_batch_size=4, gradient_accumulation_steps=4`
   → 每次 forward/backward 算 4 条样本，累积 4 次（共 16 条样本）后，再做一次梯度更新。
   → 等效 **总 batch size = 4 × 4 = 16**。

**两者关系**

- **真实 batch size** = `per_device_train_batch_size × gradient_accumulation_steps × GPU数量`
- 下表比较当两者不同设置的区别

| `per_device_train_batch_size` | `gradient_accum` | 等效 batch | 显存占用                    | 速度特点                                              |
| ----------------------------- | ---------------- | ---------- | --------------------------- | ----------------------------------------------------- |
| 2                             | 8                | 16         | **低**（每次只装 2 个样本） | **慢**（要累积 8 次 forward/backward 才更新一次参数） |
| 4                             | 4                | 16         | **高**（一次装 4 个样本）   | **快**（只累积 4 次，更新更频繁）                     |

这里可能会让人感到困惑，如何理解这里的梯度累积，它是在哪里又怎么影响梯度计算的？我们结合一个例子来理解这个过程：

>假设我们在训练一个模型，Batch Size = 4，Accumulation = 4（也就是等效 Batch Size = 16）。现在的权重 $W = 10$。学习率 $lr = 0.1$。我们要进行一轮完整的更新（处理 16 条数据），这会被拆分成 4 个小步骤（Micro-steps）。
>
>**第 1 小步（Micro-step 1/4）**
>
>- **动作**：放入第 1-4 条数据。
>- **Forward**：算出 Loss，但是**要除以Accumulation = 4**。
>- **Backward**：假设计算出这 4 条数据认为 $W$ 应该增加 2 (梯度 $g_1 = 2$)。
>- **显存里发生了什么**：
>  - $W$ 的值：**依然是 10**（因为要累积梯度，没更新）。
>  - $W$ `.grad`：0 + 2/4 = **0.5**。
>- **更新吗？**：不更新。
>
>**第 2 小步（Micro-step 2/4）**
>
>- **动作**：放入第 5-8 条数据。
>- **Forward**：算出 Loss。
>- **Backward**：计算出这 4 条数据认为 $W$ 应该增加 1 (梯度 $g_2 = 1$)。
>- **显存里发生了什么**：
>  - $W$ 的值：**依然是 10**（还是没动）。
>  - $W$  `.grad`：0.5 (旧的) + 1 (新的) / 4 = **0.75**。
>  - *注意：这里发生了“累积”，其实就是简单的加法。*
>- **更新吗？**：不更新。
>
>**第 3 小步（Micro-step 3/4）**
>
>- 以此类推，梯度`.grad`：0.75 + (-1)/4 = **0.5**。
>- **更新吗？**：不更新。
>
>**第 4 小步（Micro-step 4/4）—— 关键**
>
>- **动作**：放入第 13-16 条数据。
>
>- **Forward**：算出 Loss。
>
>- **Backward**：计算出这 4 条数据认为 $W$ 应该增加 3 (梯度 $g_4 = 3$)。
>
>- **显存里发生了什么**：
>
>  - $W$  `.grad`：0.5 + 3 / 4 = **1.25**。
>  - 此时，背包里装的是 16 条数据总共的梯度总和。
>
>- **更新吗？**：更新！现在满足了 `Accumulation = 4`，触发更新机制：
>
>  1. 修改权重：
>
>  $$W_{new} = W_{old} - (lr \times \text{平均梯度})$$ 即
>
>  $$W_{new} = 10 - (0.1 \times 1.25) = 9.875$$
>
>  2. 清空背包：把 $W$ 的 `.grad` 归零，准备迎接下一个 16 条数据。
>
>结合伪代码，我们来看一下具体操作流程：
>
>```python
># 初始化：accumulation_steps = 4
>optimizer.zero_grad() # 先清空背包
>
>for i, batch in enumerate(dataloader):
>    # --- 1. 计算 (Forward) ---
>    # 这里每次只算 batch_size=4 的数据，显存占用小
>    outputs = model(batch)
>    loss = loss_function(outputs, targets)
>    
>    # 这一步很关键！把 Loss 除以累积步数
>    # 这样累积加起来的梯度才是“平均梯度”，而不是“总和梯度”
>    loss = loss / accumulation_steps 
>    
>    # --- 2. 累积 (Backward) ---
>    # 这一步计算梯度，并自动把梯度加到 W.grad 里
>    # 此时 W 并没有被修改！
>    loss.backward() 
>    
>    # --- 3. 判断是否该更新了 ---
>    # 如果 i 是 4 的倍数 (比如第4, 8, 12...次循环)
>    if (i + 1) % accumulation_steps == 0:
>        
>        # --- 4. 更新 (Update) ---
>        # 这时候才真正修改 W 的值
>        optimizer.step() 
>        
>        # --- 5. 清理 ---
>        # 修改完 W 后，把背包清空，为下一轮累积做准备
>        optimizer.zero_grad()
>```

### 4. 微调以后，为什么模型丧失了多种语言的能力？

1. **数据分布单一**
   - 在 SFT 时只用中文（或特定任务）数据集。
   - 由于训练步骤和学习率足够大，模型参数被强烈调整 → **原本的英文、多语言能力被覆盖**。
2. **没有做 loss mask 保留原能力**
   - 如果 system / user / assistant 全部记入 loss，模型就会“习惯性遗忘”原有对话模板，学习新模式。
3. **缺少混合训练**
   - 工业界常见做法是：
     - 保留一部分 **原始 instruction-tuning 数据（多语言、通用问答）**
     - 再混合我们的任务数据一起训练（称为 **multi-task SFT**）。
   - 如果只喂进单语数据，等于是强行“单语迁移”，模型会失去泛化。
4. **步数少但影响大**
   - 即便我们只训了几十/几百步，学习率高（2e-4）+ LoRA 更新，也足够把模型分布往单语/单任务推走。

**如何解决 / 减轻**

1. **混合数据集**
   - 在微调数据里混入一部分英文/多语言 instruction 数据（哪怕只占 10-20%），保持多语言能力。
2. **降低学习率**
   - QLoRA 的 LoRA 层可以用 `1e-4` → `5e-5`，让模型保守地更新。
3. **正则化方法**
   - 使用 **KL-divergence regularization** 或 **Replay（重放原始数据）**，防止遗忘。鉴于我们没有原始数据，Replay可能不太会考虑。
4. **多阶段训练**
   - 先做轻量多语言 + 任务混合 SFT，保证语言能力；
   - 再小步数、低学习率地做 domain-specific 调优。
5. **评测时可用 LoRA 切换**
   - 不同任务/语言可以加载不同 LoRA adapter → 避免互相覆盖。