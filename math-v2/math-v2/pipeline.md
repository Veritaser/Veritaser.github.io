这是一个基于 **DeepSeekMath-V2** 论文核心思想（Self-Verifiable Reasoning），专门针对 **Qwen-1.5B** 模型在 **GSM8K** 数据集上进行优化的技术落地 Pipeline。

该方案将论文中复杂的迭代训练简化为适合小模型的 **SFT (Distillation) -> RL (GRPO)** 路径。

```markdown
# Pipeline: Tuning Qwen-1.5B for Self-Verifiable Mathematical Reasoning
> Implementation of DeepSeekMath-V2 methodology for GSM8K on Small Language Models (SLMs).

## 📋 概览 (Overview)
本 Pipeline 旨在将 **Qwen-1.5B** 转化为具备**“解题 + 自我验证”**能力的模型。
*   **输入**: 数学问题 (GSM8K)
*   **输出**: 详细解题过程 -> 逐步自我反思 -> 自评置信度分数 -> 最终答案
*   **核心方法**: Knowledge Distillation (Teacher) + GRPO (Reinforcement Learning)

### 🛠️ 环境依赖 (Prerequisites)
*   **Hardware**: 1x RTX 3090 (24GB) or 4090 (24GB) is sufficient for 1.5B.
*   **Base Model**: `Qwen/Qwen1.5-1.5B-Chat` or `Base`
*   **Dataset**: `GSM8K` (Train split)
*   **Tools**:
    *   `vllm` (for fast data generation)
    *   `llama-factory` (for SFT)
    *   `trl` / `openrlhf` (for GRPO/PPO)

---

## Phase 1: 数据构建 (Data Construction)
**目标**: 利用强模型（Teacher）生成包含“自我验证”过程的高质量数据。论文中提到，冷启动数据对小模型至关重要。

### 1.1 准备 Prompt
我们需要 Teacher 模型不仅输出答案，还要输出 `Self Evaluation` 和 `Score`。

```python
# Teacher Prompt Template
TEACHER_PROMPT = """
Role: You are a rigorous mathematics expert.

Task:
1. Solve the following math problem step-by-step.
2. After the solution, create a section titled "## Self Evaluation".
3. In the evaluation, verify your logic and calculations step-by-step.
4. Finally, assign a confidence score between 0.0 and 1.0 inside a box.

Format Rules:
## Solution
[Detailed Chain-of-Thought]

## Self Evaluation
[Verification process: Check calculation, logic, and question alignment]

Score: \boxed{1.0}

Input Problem:
{question}
"""
```

### 1.2 数据合成 (Data Synthesis)
*   **Source**: GSM8K Train Set (7.5k examples).
*   **Teacher Model**: GPT-4o, Claude-3.5-Sonnet, or DeepSeek-V3.
*   **Action**: 对每个问题调用 Teacher 模型生成回复。

### 1.3 数据清洗 (Data Cleaning)
DeepSeekMath 强调数据的准确性。
1.  **Extract Answer**: 从 Teacher 的输出中提取最终数值答案。
2.  **Verify**: 将提取的答案与 GSM8K 的 Gold Answer 对比。
    *   如果 **Match**: 保留数据，标记 `Score=1.0`。
    *   *高级策略 (Optional)*: 如果你有资源，可以特意保留一些 Teacher 答错（但 Teacher 自评发现了错误并给低分）的数据，这有助于模型学习“诚实”，但对 1.5B 模型，建议先只用正确数据做 SFT。
3.  **Format**: 转换为 SFT 训练格式 (ShareGPT/Alpaca format)。

```json
// example_train.jsonl
{
  "messages": [
    {"role": "user", "content": "Natalia sold clips..."},
    {"role": "assistant", "content": "## Solution\nFirst, ...\n## Self Evaluation\nChecking the multiplication...\nScore: \\boxed{1.0}"}
  ]
}
```

---

## Phase 2: 监督微调 (SFT)
**目标**: 让 Qwen-1.5B 学会输出 `## Solution` 和 `## Self Evaluation` 的格式，并内化基础推理能力。

### 2.1 训练配置
使用 **LLaMA-Factory** 进行全量微调 (Full Fine-tuning)。1.5B 参数很小，不需要 LoRA，全量效果更好。

*   **Learning Rate**: 2e-5
*   **Epochs**: 3
*   **Batch Size**: 4 (Gradient Accumulation 4) -> Global BS 16
*   **Max Length**: 2048 (GSM8K usually fits in <1k tokens, but verification adds length)
*   **Loss Mask**: 确保只计算 Assistant 回复部分的 Loss。

### 2.2 验证 (Sanity Check)
训练后，输入一个未见过的问题，检查模型是否会自动输出 `## Self Evaluation` 章节。如果能稳定输出格式，进入下一阶段。

---

## Phase 3: 强化学习 (RL / GRPO)
**目标**: 提升模型的准确率（Correctness）和诚实度（Faithfulness）。
**核心算法**: **GRPO (Group Relative Policy Optimization)**。相比 PPO，它不需要 Value Network，节省显存且对数学推理任务更稳定。

### 3.1 奖励函数设计 (Reward Engineering)
根据论文公式 (5) 和 (6)，我们需要设计组合 Reward。

```python
def reward_function(prompts, completions, answer_truth, **kwargs):
    rewards = []
    for completion, gold_ans in zip(completions, answer_truth):
        
        # 1. 解析模型输出
        pred_ans = extract_answer(completion)  # 提取最终数字
        pred_score = extract_score(completion) # 提取 \boxed{0.x}
        
        # 2. 结果奖励 (Outcome Reward)
        is_correct = (pred_ans == gold_ans)
        r_outcome = 1.0 if is_correct else 0.0
        
        # 3. 格式奖励 (Format Reward)
        # 强制模型遵循结构，否则没法解析
        r_format = 0.5 if ("## Solution" in completion and "## Self Evaluation" in completion) else 0.0
        
        # 4. 一致性奖励 (Consistency Reward - DeepSeek Logic)
        # 激励模型：对的时候自信(1.0)，错的时候承认(0.0)
        # 惩罚模型：错的时候盲目自信(1.0)
        # 真实分数 S_real = 1.0 if is_correct else 0.0
        # Reward = 1 - |S_pred - S_real|
        if pred_score is not None:
            r_consistency = 1.0 - abs(pred_score - float(r_outcome))
        else:
            r_consistency = 0.0 # 没写分数的惩罚
            
        # 总奖励
        total_reward = r_outcome + 0.1 * r_format + 0.5 * r_consistency
        rewards.append(total_reward)
        
    return rewards
```

### 3.2 训练流程
1.  **Prompt**: 使用 GSM8K Train Set 的 Question。
2.  **Sampling**: 对每个 Question 采样 `G=8` 个输出。
3.  **Update**: 计算 Group Advantage 并更新 Policy Model。

---

## Phase 4: 推理与部署 (Inference & Evaluation)
**目标**: 利用训练好的“自验证”能力，通过 Test-time Compute 提升 GSM8K 准确率。

### 4.1 采样策略 (Majority Vote with Verification)
使用 **Best-of-N** 策略，但加入 Score 权重。

1.  **Input**: GSM8K Test Question.
2.  **Generate**: 设定 `Temperature=0.7`，生成 `N=4` 或 `N=8` 个完整回答。
3.  **Filter & Select**:
    *   **Priority 1**: 筛选出 `Score` 为 `1.0` 的回答。
    *   **Priority 2**: 如果有多个 1.0，对比最终答案进行投票 (Majority Voting)。
    *   **Fallback**: 如果没有 1.0 分的回答，选择 `Score` 最高的那个；或者选择 0.0 分里 CoT 最长的那个（假设想得越久越可能对）。

### 4.2 示例代码逻辑

```python
candidates = model.generate(question, n=8)
valid_solutions = []

for sol in candidates:
    score = parse_score(sol) # Extract 0.0 - 1.0
    answer = parse_answer(sol)
    if score > 0.9:
        valid_solutions.append(answer)

if valid_solutions:
    final_answer = most_common(valid_solutions)
else:
    # 模型认为自己都做错了，可以触发重试或强制输出
    final_answer = parse_answer(candidates[0]) 
```

---

## 📅 里程碑 (Milestones)

1.  **Week 1**: 完成 GSM8K 数据的 Teacher 生成与清洗 (Target: 7.5k high-quality samples).
2.  **Week 1.5**: 完成 SFT 训练。Qwen-1.5B 能够在 0-shot 下稳定输出带 Self-Eval 的格式。
3.  **Week 2**: 搭建 GRPO 环境，调试 Reward Function。进行 RL 训练。
4.  **Week 2.5**: 在 GSM8K Test 集上评估。
    *   Baseline: Qwen-1.5B 原始准确率。
    *   SFT Ver: 加入 Self-Eval 后的准确率。
    *   RL Ver + Best-of-N: 最终准确率。

## ⚠️ 常见陷阱 (Pitfalls)

*   **幻觉自评**: 1.5B 模型可能学会了“只要我给 1.0 分，Reward 就高”。**解决方法**: 必须严格执行 $R_{consistency}$，如果你做错了还打 1.0 分，必须给负奖励或零奖励，强迫它在做不出来时承认错误。
*   **格式崩坏**: 小模型在 RL 阶段容易遗忘格式。**解决方法**: 在 Reward 中保留较重的 Format Reward，或者在 GRPO 中混入少量的 SFT 数据进行正则化。
*   **计算开销**: 如果 Teacher 模型太贵，可以先用 1k 条数据做实验，验证流程通了再跑全量。
```