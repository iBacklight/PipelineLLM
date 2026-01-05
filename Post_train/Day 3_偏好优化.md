# Day 3  偏好优化

[TOC]



## 1. 偏好学习到底在学什么？

**任务**：给定相同输入 $x$ 的两段候选输出 $y^+$（更好）与 $y^-$（更差），学一个打分器或策略，让模型更倾向于产出 $y^+$。

### 两种等价视角[1] [2]：

1. **打分器视角（score-based）**
    学一个 $s_\theta(x,y)$，压低损失

$$
\mathcal L = -\log \sigma\!\big(s_\theta(x,y^+) - s_\theta(x,y^-)\big),
$$

这里的 $\sigma$ 指的是 **sigmoid 逻辑函数**。数学形式是：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
意思就是“让优的比分差更大”。这就是经典的**对数几率（logistic）pairwise loss**。

2. **选择概率视角（choice-probability）**
    假设“效用”=可观测得分+噪声，$\Pr(y^+ \succ y^-)=\sigma\big(s_\theta(x,y^+)-s_\theta(x,y^-)\big)$。

- 噪声若服从**Gumbel**：得到 **Bradley–Terry ** 家族；
- 噪声若服从**高斯**：得到 **Thurstone–Mosteller**（probit）模型。

> 小结：两种视角最后都落在“**学一个能把好样本的分数拉高于差样本**”的目标上。

### 两个直观的例子

1. **例子 A：电影推荐（非 LLM）**
    输入 $x$=用户画像，候选 $y^+$=《星际穿越》，$y^-$=《电影B》。当前打分 $s(y^+)=2.0, s(y^-)=1.2$，那么
    $\Pr(y^+\succ y^-)=\sigma(0.8)\approx 0.69$。训练就让这个概率更接近 1：要么拉高 $s(y^+)$，要么压低 $s(y^-)$，或两者都来点。

2. **例子 B：LLM 回答偏好**
   输入 $x$=“请给一个能直接运行的 Python 例子并解释复杂度”。 $y^+$=结构清晰、可运行、解释到位；$y^-$=解释含糊、代码跑不通。我们希望策略 $\pi_\theta$生成 $y^+$ 的对数几率更大, 即：

$$
\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) \uparrow
$$

​        这就是 DPO 的核心推动力，常常被用于推荐系统上。

---

## 2. 针对于LLM 的偏好和对齐方法

- **RM（Reward Model）pairwise 训练**：用 BT logistic 拟合“$y^+$ 优于 $y^-$”；再用 PPO 等 RL 优化策略。
- **DPO / IPO / ORPO / KTO（偏好到监督）**：绕开显式 RM，直接用 pairwise 偏好更新策略分布；
  - **DPO**：最常用，稳定、实现简单；
  - **IPO/KTO/ORPO**：对损失形式、鲁棒性、噪声建模做不同取舍。
- **GRPO/GSPO（组内相对）**：同一 prompt 多样本内部做“相对”加权，介于 pairwise 与 RL 之间（Day 4）。

---

## 3. 直接偏好优化（DPO）

### 3.1 DPO简介

在对齐里，我们常有“成对偏好”数据：同一输入 $x$ 下，两段回答 $y^+$（更好）和 $y^-$（更差）。

- **目标**：让策略 $\pi_\theta$ 更倾向产生 $y^+$ 而不是 $y^-$，且又不要**偏离参考策略**（通常是SFT模型）太远。

传统 RLHF的做法是，先训奖励模型（RM）拟合偏好， 再用 PPO 等 RL 优化策略 + KL 正则。

**DPO**（Direct Preference Optimization）则跳过显式 RM，直接把偏好约束写成一个**监督式目标**来优化策略，训练稳定、便宜、易落地。换个说法，DPO是 从“最大化奖励–KL”的RL目标出发，通过数学变换把“**奖励差**”用**参考策略的对数几率差**替代，得到一个对**偏好对**的**二元逻辑回归**损失。于是我们无需显式的RM，可以直接对策略做**pairwise**监督优化。对一个样本 $(x, y^+, y^-)$，DPO 的 loss 常写作：
$$
\mathcal{L}_{\text{DPO}}(\theta)
= -\log \sigma\!\big(
\beta\,[\,\underbrace{\log \pi_\theta(y^+|x)-\log \pi_\theta(y^-|x)}_{\text{策略对数几率差}}
-\underbrace{\log \pi_{\text{ref}}(y^+|x)+\log \pi_{\text{ref}}(y^-|x)}_{\text{参考校正/隐式KL}}]\big).
$$

- $\sigma(\cdot)$ 是 sigmoid，把实数映射为“$y^+$ 胜出”的概率。
- $\pi_{\text{ref}}$：参考策略（通常是 Day 2 的 SFT 模型，**冻结**）。
- $\beta>0$：**KL 控制力度**的温度系数。大一点→更守旧（更贴近参考、不易啰嗦），小一点→更敢变化。

直观理解：

- 如果策略对 $y^+$ 的对数概率远高于 $y^-$（且不偏离参考太多），loss 会很小；
- 若策略对 $y^-$ 更“偏爱”，或相对参考偏差过大，loss 会增大，推动更新。

说白了，**DPO就是一个把模型对齐偏好的过程，就是一个对齐（Alignment）的方法。注意：DPO的过程不是在自回归采样，而是输入response（target）进行“评分（teacher forcing）”**。

### 3.2 DPO 深度理解加公式推导

#### 3.2.1 前置概念：KL散度

**KL 散度（Kullback–Leibler divergence**衡量两个分布 $p,q$ 的差异：
$$
D_{\mathrm{KL}}(p\Vert q)=\sum_{y} p(y)\,\log\frac{p(y)}{q(y)} \quad(\text{或 } \mathbb{E}_{y\sim p}[\log p(y)-\log q(y)])
$$
他表示p分布对于q分布的相似程度

- $D_{\mathrm{KL}}\ge 0$（Jensen不等式），且当且仅当 $p=q$ 时为 0， 越接近0，表明他们越相似。
- 它是**有方向的**，不对称的：$D_{\mathrm{KL}}(p\|q)\neq D_{\mathrm{KL}}(q\|p)$。
- 在对齐里，KL 常用来**“拽回去”**：不让新策略 $\pi$ 偏离“参考策略” $\pi_{\text{ref}}$ 太远（比如 SFT 模型），避免输出风格/安全性跑飞。

在RLHF中（下一节），KL散度常被用于正则化策略上。例如，在每个输入 $x$ 上，我们希望策略 $\pi(\cdot|x)$ **既能拿到高“奖励” $r(x,y)$，又别离参考 $\pi_{\text{ref}}(\cdot|x)$ 太远**。一个经典的一步式目标是（类似PPO-KL范式）：
$$
\max_{\pi(\cdot|x)}\;\; \mathbb{E}_{y\sim \pi(\cdot|x)}[\,r(x,y)\,]\;-\;\beta\;D_{\mathrm{KL}}\!\left(\pi(\cdot|x)\,\big\Vert\,\pi_{\text{ref}}(\cdot|x)\right) =\\
\min_{\pi(\cdot|x)}\;\; \beta\;D_{\mathrm{KL}}\!\left(\pi(\cdot|x)\,\big\Vert\,\pi_{\text{ref}}(\cdot|x)-\mathbb{E}_{y\sim \pi(\cdot|x)}[\,r(x,y)\,]\;\;)\right.
\tag{1}
$$

>论文中写道 [3]： During the RL phase, the learned reward function is used to provide feedback to the language model. Following prior works [17, 18], the optimization is formulated as (1)

这里 $\beta>0$ 控制“保守 vs 激进”。$\beta$​ 大：更贴近参考；小：更敢改变，远离参考。这里，对给定的 $x$，
$$
D_{\mathrm{KL}}\!\big(\pi(\cdot|x)\,\|\,\pi_{\rm ref}(\cdot|x)\big)
=\mathbb{E}_{y\sim \pi(\cdot|x)}
\!\left[\log \frac{\pi(y|x)}{\pi_{\rm ref}(y|x)}\right].
$$


#### 3.3.2 前置概念：BT偏好数据模型

然而现实里，我们很难有显式 $r(x,y)$，但通常有**偏好对数据** $(x,y^+,y^-)$，即对同一 $x$，标注者认为“$y^+$ 胜过 $y^-$”。这时候可以使用一个常用模型——***Bradley–Terry Model***[1]，该公式原文有详细讲解[3]:
$$
\Pr(y^+ \succ y^- \mid x)\;=\;\sigma\!\big(r(x,y^+)-r(x,y^-)\big),
$$
这是说，我们规定：

- 每个候选输出 $y$ 有一个“效用/得分” $r(x,y)$。
- 如果两个候选 $y^+, y^-$ 拿来比较，我们假设它们的效用差 $r(x,y^+) - r(x,y^-)$ 决定了哪一个更好。
- 为了把差值映射到 $[0,1]$ 之间的概率，用 sigmoid 函数。

这个就是 pairwise logistic preference learning 的**标准写法**。其实，如果我们结合**SFT中交叉熵对于最大似然函数**的应用，就能更直观地理解(可选了解)：

1. **最大似然（MLE）目标**： 给定标注者的观察结果“$y^+$ 胜过 $y^-$”，我们希望最大化模型对该事件的概率：
   $$
   \max_\theta \;\;\log \Pr_\theta(y^+ \succ y^- \mid x)
   = \max_\theta \;\;\log \sigma(r_\theta(x,y^+) - r_\theta(x,y^-)).
   $$

2. **对数似然转损失**：取负号就是训练损失：
   $$
   \mathcal{L}(\theta) = -\log \sigma(r_\theta(x,y^+) - r_\theta(x,y^-)).
   $$

3. **与交叉熵的等价性**：注意，这其实就是**二分类的交叉熵损失**：我们要区分“$y^+$ 优于 $y^-$”（标签=1）与“相反情况”（标签=0）。如果写成通用形式：
   $$
   \mathcal{L}(\theta) = \text{CE}\Big(\sigma(r_\theta(x,y^+)-r_\theta(x,y^-)), \;1\Big),
   $$
   其中 CE 是二元交叉熵：
   $$
   \mathcal{L}_{CE}(y, \hat{y}) = -\big[ y \log \hat{y} + (1-y)\log(1-\hat{y}) \big].
   $$
   这里就可以理解， $\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$相当于**在输入 $x$ 下，模型判定候选输出 $y^+$ 优于 $y^-$ 的概率**。这个值越接近1越好。（因为是概率，所以最大就是1）

#### 3.3.3 DPO loss 公式推导

我们接下来正式进行推导。从原文中看，作者从强化学习微调阶段的目标函数（1）式中，分析得出在 KL 约束条件下，最大化奖励的**最优策略形状**（对每个固定的 $x$）：
$$
\pi_\tau(y\mid x)
=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,
\exp\!\Big(\tfrac{1}{\beta} \, r(x,y)\Big),
\tag{2}
$$
其中 $Z(x)=\sum_y \pi_{\text{ref}}(y\mid x)\exp(\tfrac{1}{\beta}r(x,y))$ 是配分函数(partition function)， 该项只随 $x$ 变，不随 $y$ 变，是归一化函数。为了让乘法变成加法，使得容易做**差分**消去配分函数同时更加稳定，两边取对数并移项，得到：
$$
r(x,y) = \beta \log \frac{\pi_\tau(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x).
\tag{3}
$$
注意 $\log Z(x)$ **只依赖于 $x$**，与 $y$​ 无关。然后，Bradley–Terry（BT）假设对同一 $x$ 的两段输出 $y_1,y_2$：
$$
p^*(y_1 \succ y_2 \mid x) \;=\; \sigma\!\big(r^*(x,y_1) - r^*(x,y_2)\big).
\tag{4}
$$
把 (3) 代入到“奖励差”里（$r^*$ 用最优策略 $\pi^*$ 的表达），$\beta\log Z(x)$ 在做差时**抵消**，得到（由此可知配分函数被消除后对梯度也没有影响）：
$$
p^*(y_1 \succ y_2 \mid x)
= \sigma\!\Big(
\beta \log \frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}
-\beta \log \frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}
\Big).
\tag{5}
$$
把 $\sigma(z)=1/(1+e^{-z})$ 展开，就得到原文等价写法：
$$
p^*(y_1 \succ y_2 \mid x)=
\frac{1}{1+\exp\!\Big(\beta \log \frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}
-\beta \log \frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\Big)}.
\tag{6}
$$
用 $\pi_\theta$ 近似 $\pi^*$，提取$\beta$，再对每个偏好对 $(x,y_w,y_l)$ 的“标签=1（$y_w$ 赢）”做伯努利极大似然：
$$
\max_\theta\;\;\log\sigma\!\Big(\beta\big[\underbrace{\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{边际}}\big]\Big).
$$
取负号，变为求min的最小值，就是训练时用的 **DPO 损失**：
$$
\mathcal L_{\text{DPO}}(\theta;\pi_{\text{ref}})
= -\,
\Big[\log \sigma\!\Big(
\beta \big(\underbrace{\log \pi_\theta(y_w\mid x)-\log \pi_\theta(y_l\mid x)}_{\text{策略对数几率差}}
-\underbrace{\log \pi_{\text{ref}}(y_w\mid x)+\log \pi_{\text{ref}}(y_l\mid x)}_{\text{参考校正}}\big)
\Big)\Big].
$$
Again，**$\beta$ 大：更贴近参考；小：更敢改变，远离参考**。总结，DPO的梯度更新旨在**增加**优质回答的概率，同时**减少**劣质回答的概率。

#### 3.3.4 和SFT/RLHF的差异

| 方法           | 数据需求                       | 是否显式RM | 是否环境交互 | 优点                             | 风险/局限                         |
| -------------- | ------------------------------ | ---------- | ------------ | -------------------------------- | --------------------------------- |
| **SFT**        | 带“好示例”的监督数据           | 否         | 否           | 稳定、便宜                       | 只能“学像”，不能“学偏好”          |
| **DPO**        | 成对偏好（chosen vs rejected） | 否         | 否           | 直接优化偏好，训练简单稳定       | 依赖偏好覆盖面；易出现“变长/啰嗦” |
| **PPO (RLHF)** | 偏好→RM→奖励                   | 是         | 是           | 能直接最大化奖励（可自定义指标） | 工程复杂、昂贵、易不稳            |
| **GRPO/GSPO**  | 组内多样本相对偏好             | 否         | 可无         | 折中稳定，弱化对RM依赖           | 需多样本采样，仍有实现细节        |

---

## 4. DPO 训练复现

我们使用`torch`和`transformer`作了最小可执行的DPO算法，在这里我们列出loss计算的流程。对于测试阶段，我们的俄数据集需要包含正向偏好和负向偏好，类似于：

```python
{"prompt": "What is the capital of France?", 
 "chosen": "The capital of France is Paris. Paris is located in the north-central part of             the country and is known for its rich history, culture, and landmarks like the             Eiffel Tower.", 
 "rejected": "I don't know the capital of France."}
```

简洁地，dpo的loss可以被写成：

```python
"""
Compute DPO loss.

Args:
    pol_pos_lp: Policy model log prob for positive examples
    pol_neg_lp: Policy model log prob for negative examples
    ref_pos_lp: Reference model log prob for positive examples
    ref_neg_lp: Reference model log prob for negative examples
    beta: DPO temperature parameter
"""
# Δ = (logπθ(y+)-logπθ(y-)) - (logπref(y+)-logπref(y-))
delta = (pol_pos_lp - pol_neg_lp) - (ref_pos_lp - ref_neg_lp)
# -log σ(βΔ) = softplus(-βΔ)
return F.softplus(-beta * delta).mean()
```

这里使用了[softplus](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html)函数，因为：
$$
-\log \sigma(z) = \log(1+e^{-z}) = \text{softplus}(-z).\\
$$
这种写法在数值上比 `-torch.logsigmoid(z)` 更稳定，尤其是当 $\beta\Delta$ 取极大/极小值时，能避免溢出。然后，训练时我们通常一个 batch 有多条样本：$\mathcal L = \frac{1}{N} \sum_{i=1}^N -\log \sigma(\beta \Delta_i).$ 所以 `mean()` 就是做 **batch 平均**。如果不取 mean， 会只返回一个向量，梯度就会按样本数放大，训练不稳定。

接下来我们来walkthrough训练过程，这轮训练的输入与目标：

- 每个 batch 含成对样本：`(x, y+ , y-)`。
- 我们要让**策略**相对**参考**更偏爱 `y+`（chosen）而不是 `y-`（rejected）。
- 损失：$\mathcal L_{\text{DPO}}=-\log \sigma(\beta\Delta)$

**1）取 batch **

```python
pos_ids = batch.pos_input_ids.to(device) # prompt + response(target)在词表中的token id 
neg_ids = batch.neg_input_ids.to(device)
pos_mask_tok = batch.pos_mask.to(device) # label mask
neg_mask_tok = batch.neg_mask.to(device)
```

- `pos_ids/neg_ids`：把 **prompt+response** 拼好的 token 序列（要和SFT的tokenizer一致）。
- `pos_mask_tok/neg_mask_tok`：**标签掩码（label mask）**，在 **response 段**为 1，prompt 段和 padding 为 0。
  - 作用：后面算序列对数似然时，只累计 **response token** 的 logprob（这与 SFT 的 “assistant mask” 一致）。
  - 这是 **DPO 的关键**：我们比较的是“同一 prompt 下，两段完整回答的序列对数似然差”，因此要把 prompt 的 token 排除在外。在具体做法可以见代码，或这看接下来的实战。

**2) attention mask（给 Transformer 用）**

```python
pos_attn = (pos_ids != tokenizer.pad_token_id).long()
neg_attn = (neg_ids != tokenizer.pad_token_id).long()
```

- **attention mask** 告诉模型哪些位置是 **有效 token**（1），哪些是 **pad**（0）。和上面的 **label mask** 不同，这是给模型注意力用，屏蔽 padding；**prompt+response 都是 1**。

**3) autocast（混合精度）与两套前向**

```python
# autocast：开启 bf16/fp16 混合精度，省显存提速。
with torch.amp.autocast(device_type="cuda", dtype=config["dtype"], enabled=True):
    pol_pos_logits = forward_policy(policy, pos_ids, pos_attn)
    pol_neg_logits = forward_policy(policy, neg_ids, neg_attn)

    with torch.no_grad():# 参考策略不参与反传
        ref_pos_logits = forward_ref(ref, pos_ids, pos_attn)
        ref_neg_logits = forward_ref(ref, neg_ids, neg_attn)
```

- **policy**：可训练的策略；**ref**：**冻结**的参考策略（一般是 Day2 的 SFT 模型）。
- 对每个 `(x, y+)` 和 `(x, y-)` 分别走一次前向，得到 `[B, T, V]` 的 `logits`。（B-Batch size, T-seq len（每个样本 token 序列的长度）， V-vocab size（模型词表的大小））。
- 注意！：虽然这里的 `pos_ids`/`neg_ids` 都是 **[prompt || response]** 的拼接，但这不是“让模型生成”，而是让模型**打分**。也就是说，我们需要模型在**已知整段目标响应**的前提下，逐 token 计算“生成这一 token 的概率”。这就是**teacher forcing / 评分模式**，不是自回归采样。
  - 为了得到这条序列的 logprob，我们必须把 **prompt 与（作为监督目标的）response** 一起输入；随后模型前向得到的是 `logits`（未归一化分数）；（见下面几步）我们再 `log_softmax` + `gather` + 聚合，才得到“这条 response 的（对数）概率”。

**4) 从 logits 得到“序列对数似然”**

```python
pol_pos_lp = seq_logprob_from_logits(
    pol_pos_logits[:, :-1], pos_ids[:, 1:], pos_mask_tok[:, 1:], config["length_norm"]
)
...
# seq_logprob_from_logits
def seq_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor,mask: torch.Tensor, 
    length_norm: bool
) -> torch.Tensor:
    """
    Compute sequence log probabilities from logits.
    
    Args:
        logits: [B, T, V] - model logits
        labels: [B, T] - target token ids
        mask: [B, T] - mask for response tokens only
        length_norm: whether to normalize by length
        
    Returns:
        [B] - sequence log probabilities
    """
    # 在最后一个维度（词表维度）做 softmax，再取对数 → 得到每个位置对所有 token 的 log 概率。
    log_prob = F.log_softmax(logits, dim=-1)
    # 从完整的 vocab 概率分布里，只保留“预测真实 token”那一项。
    tok_log_prob = log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    tok_log_prob = tok_log_prob * mask # 相乘后，prompt 和 pad 的 log 概率会被置零，只保留 response 部分。
    
    if length_norm: # 把总 logprob 除以长度 ∣𝑦∣， 即取平均：
        denom = mask.sum(-1).clamp_min(1.0)
        return tok_log_prob.sum(-1) / denom
    else:
        return tok_log_prob.sum(-1)
```

这里有三个关键点：

**(a) “shift” 一格（next-token 预测）**，语言模型是 **预测下一个 token**：

- 用 `logits[:, :-1]` 预测 `labels = input_ids[:, 1:]`。
- 这就是常说的 **teacher forcing**（和 HuggingFace `labels` 移位一致）。

**(b) label mask 只数 response token**，`pos_mask_tok[:, 1:]`： prompt 段与 pad 都是 0，**response 段是 1**。

**(c) policy 与 ref 两套都要算**

- 得到四个标量向量（按 batch）：`pol_pos_lp, pol_neg_lp, ref_pos_lp, ref_neg_lp`。
- 这些就是 $\log\pi_\theta(y^+|x), \log\pi_\theta(y^-|x)$ 及对应参考项。

这里额外说一下以下代码：

```python
#                              dim         index
tok_log_prob = log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1) 
```

这里gather的作用是沿着log_prob的dim，按照index抽取向量。结果返回和 `index` 的形状一样，每个位置存的是 `log_prob` 对应下标的值。在这个场景中，logp` 的形状 `[B, T, V]， labels` 的形状 `[B, T]，里面是**真实 token 的索引**（每个位置一个整数）。我们回到前面的提取的部分，提取的时候

```python
pol_pos_logits[:, :-1]	 # shape [B, T-1, V]，扔掉最后一个位置的预测（因为没有“下一个token”可以对齐）。
pos_ids[:, 1:]           # shape [B, T-1]，扔掉第一个token（因为第0个logit预测的就是token1）。
pos_mask_tok[:, 1:]      # shape [B, T-1]，同样扔掉第一个token的mask，保证和labels对齐。
```

**5）计算 DPO 的 margin 与 loss**

```python
# Compute DPO loss
loss = dpo_loss(pol_pos_lp, pol_neg_lp, ref_pos_lp, ref_neg_lp, config["beta"]) / config["grad_accum"]
```

**6） 反传与优化**

```python
scaler.scale(loss).backward()
if (batch_idx + 1) % grad_accum == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

- `GradScaler`：配合 autocast 的 **混合精度**，避免 fp16 梯度下溢。
- **梯度累积**：每 `grad_accum` 个小步做一次优化器 step，相当于更大的有效 batch（省显存）。
- `scheduler.step()`：学习率调度（线性 warmup + 线性 decay 等）。

到此为止，我们就可以进行简单的DPO部署了。在接下来的实战中，我们可能会用到unsloth和trl的官方库，而不会使用该自定义的DPO。

---

## 5. DPO 实战

有了上面的经验，我们针对SFT使用过的Qwen3 4b模型继续使用DPO进行优化，我们将使用我们自定义的DPO和工业库跑一遍相同的数据集，以对比他们的情况。

### 5.1 数据集

目标统一成三列格式：

```
prompt | chosen | rejected
```

这里推荐的起步的dataset，包含中英文，防止在SFT中出现的模型失语的能力：

- UltraFeedback/HelpSteer2（偏好或打分可转 pair；只选高置信度样本）[4]

  - 其评价指标多元，我们可以对给出的评价指标**先做归一化**再**加权求和**，推荐权重：`correctness 0.4, helpfulness 0.3, coherence 0.2, 剩余复杂度和冗长度各位0.5`（理由：先确保“对”，再考虑“有用”，最后“连贯”）。

- opencsg/UltraFeedback-chinese [5]（中文偏好打分数据集，含有Deepseek进行的多元打分，可生成偏好对）

  - 有相似问题。我们先**按相同评分规则(权重分配)做归一化**、剔除异常短/模板化回答，并确保**chosen≠rejected**。

    ```python
    annotation_mapping = {
    "helpfulness": "helpfulness",
    "honesty": "correctness",  # Map honesty to correctness
    "instruction_following": "coherence",  # Map instruction to coherence
    "truthfulness": "correctness"  # Map truthfulness to correctness
    }
    ```

> 实战建议：取 30k–50k 对，覆盖 70% 英文 + 30% 中文，并获得稳定的 win-rate 信号（随后再扩容与清洗迭代）。

我们按照以下规则sample 数据集的大小：

```python
    "sampling": {
        "total_samples": 7500,  # Target total samples
        "en_ratio": 0.7,        # English ratio (70%)
        "zh_ratio": 0.3,        # Chinese ratio (30%)
        "en_samples": 5250,     # English samples (70% of 7500)
        "zh_samples": 2250      # Chinese samples (30% of 7500)
    }
```

其他对数据集的处理，详见`dpo/dataset/data_process.py`。

### 5.2 主流库选取

| 架构/工具                           | 上手难度 | 算法覆盖                  | 工程特性                                   | 4080 友好度              | 适合本轮吗？               |
| ----------------------------------- | -------- | ------------------------- | ------------------------------------------ | ------------------------ | -------------------------- |
| **TRL (HuggingFace)**               | 低       | DPO/IPO/ORPO/SimPO/KTO 等 | 与 Transformers/PEFT/Accelerate 原生协同好 | 高（单卡、QLoRA 非常稳） | ✅ 首选跑通基线             |
| **LLaMA-Factory**                   | 低-中    | DPO/ORPO/KTO…             | CLI 配置即跑，多数据格式适配               | 高                       | 作为“命令行版”快速复现     |
| **OpenRLHF / VE**（工程化 RL 框架） | 中-高    | DPO/GRPO/PPO…             | 多机多卡、流水线、吞吐强                   | 中（单卡略重）           | 了解工程形态，用于扩展     |
| **自研轻框架（TRL 上包一层）**      | 中       | 灵活                      | 完全可控、便于做 CL 循环接口               | 高                       | ✅ 便于后续 Day5/6 循环实验 |

我们选用TRL框架搭配unsloth(`train_dpo.py`)，与自己写的客制化dpo项目(`mini_dpo.py`), 同时进行本次实验，以对比性能。

### 5.3 核心参数

**训练配置（与 SFT 对齐）**

- **batch_size = 1**：单卡显存紧张时用最小批；配合梯度累积保持有效批量。
- **grad_accum = 24**：累积 24 次再更新一次参数，相当于有效批量 = 1×24，曲线更稳。
- **learning_rate = 1e-5**：DPO+QLoRA 的保守学习率，降低抖动与过拟合风险。
- **epochs = 1**：先跑 1 个 epoch 观察指标（win-rate/合法率），再决定是否继续。
- **max_length = 1024**：限制拼接后的序列长度，节省显存与训练时间。
- **beta = 0.1**：DPO 的偏好强度；0.1 是稳健起点，过大易抖、过小改进慢。
- **length_norm = True**：按长度归一化对数似然，显著减少因长短差异引起的 loss 波动。
- **use_lora = PEFT_AVAILABLE**：若环境装有 PEFT 就启用 LoRA，仅训练少量增量参数，省显存。

**量化（BitsAndBytes，QLoRA）**

- **load_in_4bit = True**：以 4-bit 权重量化加载底座模型，大幅降低显存占用。
- **bnb_4bit_quant_type = "nf4"**：NF4 量化格式，相比 q4_0 通常精度更好。
- **bnb_4bit_compute_dtype = config["dtype"]**：计算精度（建议 bf16）；权重是 4-bit，计算用 bf16/FP16。
- **bnb_4bit_use_double_quant = True**：双重量化进一步压缩统计量，继续省显存、对效果影响小。

**LoRA 配置（PEFT）**

- **r = 32**：LoRA 秩；表示增量矩阵容量。r 越大容量越强但显存开销↑。
- **lora_alpha = 64**：缩放系数，常取 2×r；控制 LoRA 更新的有效强度。
- **lora_dropout = 0.01**：轻微正则，减轻过拟合；设太大会降收敛速度。
- **bias = "none"**：不训练 bias，进一步节省参数与显存。
- **task_type = "CAUSAL_LM"**：自回归语言建模任务类型，确保注入点与调度正确。
- **target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]**：在注意力与 MLP 关键投影层挂 LoRA，覆盖面广、效果稳定。

### 5.4 模型选取

我们使用`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`来做这次实验（4b显存和储存空间不够了lol， base policy model和ref model均选择该模型。

## 6.其它偏好学习策略

### 6.1 LiPO（Listwise Preference Optimization）[6]

**要点**：把对齐视为**学习排序（Learning-to-Rank）\**问题，不再把 K 个候选拆成成对比较，而是\**直接利用整组排序**做监督。给定同一提示 $x$ 的 K 个候选 $\{y_1,\dots,y_K\}$ 及其排序（或打分），令策略 $\pi_\theta$ 产出序列对数似然，按 LTR 的 **listwise** 目标优化策略参数 $\theta$（本质是“监督式偏好优化”，非 RL）。

**偏好建模**：LiPO 在统一框架下给出 pointwise / pairwise / listwise 多种目标，其中**listwise** 直接最大化“给定排序的概率”，可用 ListMLE / softmax / Lambda-loss 等实现。一个常见的 **ListMLE** 写法（对单个 $(x,\{y\},\text{perm})$）是：
$$
\mathcal{L}_{\text{ListMLE}}(\theta)
= -\sum_{t=1}^{K}\Big[
s_\theta(x,y_{(t)}) - \log\sum_{j=t}^{K} \exp\big(s_\theta(x,y_{(j)})\big)
\Big],
$$
其中 $y_{(t)}$ 表示第 $t$ 名候选，$s_\theta$ 可取序列的归一化 log-prob 或把策略-打分与 RM/评分结合的可微分数。LiPO 论文实作中同时比较了 **pair-logistic、pair-hinge、list-MLE、lambda-loss** 等多种损失，并报告 listwise 类目标在**真实 rankwise 数据**上对 DPO 等基线有增益（如 LiPO-$\lambda$）。



## Reference

[1] Bradley, R. A., & Terry, M. E. (1952). *Rank analysis of incomplete block designs: I. The method of paired comparisons*. Biometrika, 39(3/4), 324–345.

[2] Thurstone, L. L. (1927). *A law of comparative judgment*. Psychological Review, 34(4), 273.

[3] Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290.

[4] Wang, Zhilin, et al. "Helpsteer 2: Open-source dataset for training top-performing reward models." *Advances in Neural Information Processing Systems* 37 (2024): 1474-1501.

[5] https://huggingface.co/datasets/opencsg/UltraFeedback-chinese

[6] Liu, T., Qin, Z., Wu, J., Shen, J., Khalman, M., Joshi, R., Zhao, Y., Saleh, M., Baumgartner, S., Liu, J., Liu, P.J., & Wang, X. (2024). LiPO: Listwise Preference Optimization through Learning-to-Rank. *ArXiv, abs/2402.01878*.

[7] 

## Appendix

### BT 模型例子

下面用一个**Bradley–Terry（BT）成对比较模型**的例子，把公式与计算走一遍。

1. **场景**

有三位选手：A、B、C。我们做了几次两两对决，记录到：

- A 对 B：A 赢 7/10（p(A≻B)=0.7）
- A 对 C：A 赢 6/10（p(A≻C)=0.6）
- C 对 B：C 赢 6/10（p(C≻B)=0.6）

2. **BT 模型设定**

给每个选手一个“强度/偏好参数” $w_i$。A 胜过 B 的概率建模为：
$$
\Pr(A \succ B) \;=\; \sigma(w_A - w_B) \;=\; \frac{1}{1+e^{-(w_A-w_B)}}
$$
同理对其它组合。注意：**只识别强度差** $w_i-w_j$，所以整体加常数不变（我们甚至可把一个人的强度设为 0 作基准）。

**由数据“反推”强度差**

把**观测胜率当作真概率**的近似，就有（用对数最大似然估计 logit($p$)=$\ln\frac{p}{1-p}$）：
$$
\begin{aligned}
w_A - w_B &\approx \text{logit}(0.7) = \ln\!\frac{0.7}{0.3} = \ln\!\frac{7}{3} \approx 0.8473 \\
w_A - w_C &\approx \text{logit}(0.6) = \ln\!\frac{0.6}{0.4} = \ln(1.5) \approx 0.4055
\end{aligned}
$$
设定基准 $w_A=0$，则
$$
w_B \approx -0.8473,\qquad w_C \approx -0.4055.
$$
**用这个强度去“预测”另一对**

现在预测 **B 对 C** 的胜率（顺便检查一致性）：
$$
\Pr(B \succ C)=\sigma(w_B - w_C)=\sigma(-0.8473 - (-0.4055))=\sigma(-0.4418).
$$
计算 $\sigma(-0.4418)=\frac{1}{1+e^{0.4418}}\approx \frac{1}{1+1.556}\approx 0.391$。

也就是 **B 胜 C 约 39.1%**，等价地 **C 胜 B 约 60.9%**，和我们观测的 0.6 很接近。

