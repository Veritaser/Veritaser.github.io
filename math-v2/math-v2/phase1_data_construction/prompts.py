"""
Prompt Templates for Teacher Model Data Generation
用于Teacher模型生成带自我验证过程的数据
"""

# Teacher Prompt Template - 用于生成带Self Evaluation的训练数据
TEACHER_PROMPT = """Role: You are a rigorous mathematics expert.

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

Score: \\boxed{{1.0}}

Input Problem:
{question}"""

# System Prompt for Teacher Model
TEACHER_SYSTEM_PROMPT = """You are a rigorous mathematics expert who always:
1. Shows complete step-by-step reasoning
2. Verifies your own work through self-evaluation
3. Provides honest confidence scores based on your verification
4. Uses clear mathematical notation and formatting"""

# Student Model Inference Prompt
STUDENT_PROMPT = """Solve the following math problem step-by-step, then verify your solution.

Problem:
{question}"""

# 中文版本的Prompt (可选)
TEACHER_PROMPT_ZH = """角色: 你是一位严谨的数学专家。

任务:
1. 逐步解决以下数学问题。
2. 解答完成后，创建一个标题为 "## 自我评估" 的章节。
3. 在评估中，逐步验证你的逻辑和计算。
4. 最后，在方框中给出一个 0.0 到 1.0 之间的置信度分数。

格式要求:
## 解答
[详细的思维链推理]

## 自我评估
[验证过程：检查计算、逻辑和问题对齐]

分数: \\boxed{{1.0}}

输入问题:
{question}"""


def get_teacher_prompt(question: str, language: str = "en") -> str:
    """
    获取Teacher模型的prompt
    
    Args:
        question: 数学问题
        language: 语言，'en' 或 'zh'
    
    Returns:
        格式化后的prompt
    """
    if language == "zh":
        return TEACHER_PROMPT_ZH.format(question=question)
    return TEACHER_PROMPT.format(question=question)


def get_student_prompt(question: str) -> str:
    """
    获取Student模型推理时的prompt
    
    Args:
        question: 数学问题
    
    Returns:
        格式化后的prompt
    """
    return STUDENT_PROMPT.format(question=question)
