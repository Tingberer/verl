#!/usr/bin/env python3
"""
自定义规则奖励函数示例文件

这个文件包含多个规则奖励函数的实现，可以根据不同任务选择使用
配置时设置: custom_reward_function.name=函数名
"""

import re
import math
from typing import Dict, Any, Optional


def my_custom_reward_function(data_source: str, solution_str: str, ground_truth: str, 
                              extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """
    通用自定义规则奖励函数
    
    Args:
        data_source: 数据源标识 (如 "gsm8k", "math", "general_qa")
        solution_str: 模型生成的回答
        ground_truth: 标准答案
        extra_info: 额外信息字典
        **kwargs: 其他参数
        
    Returns:
        float: 奖励分数 (0.0 - 1.0)
    """
    score = 0.0
    
    # 规则1: 长度奖励 - 回答长度合理性
    response_length = len(solution_str.split())
    if 10 <= response_length <= 150:
        score += 0.2
    elif 5 <= response_length < 10:
        score += 0.1  # 稍短
    elif response_length > 150:
        score -= 0.1  # 太长扣分
    
    # 规则2: 格式奖励 - 良好的结构
    if has_good_structure(solution_str):
        score += 0.2
    
    # 规则3: 数据源特定奖励
    if data_source in ["gsm8k", "math"]:
        score += math_specific_reward(solution_str, ground_truth)
    elif data_source == "general_qa":
        score += qa_specific_reward(solution_str)
    
    # 规则4: 礼貌和专业性
    score += politeness_reward(solution_str)
    
    return min(max(score, 0.0), 1.0)  # 确保在 [0,1] 范围内


def math_specific_reward(solution_str: str, ground_truth: str) -> float:
    """数学题专用奖励函数"""
    score = 0.0
    
    # 检查是否包含计算过程
    calculation_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'=\s*\d+', r'计算', r'solve']
    if any(re.search(pattern, solution_str, re.IGNORECASE) for pattern in calculation_patterns):
        score += 0.3
    
    # 检查是否有明确的答案格式
    answer_patterns = [r'答案.*?(\d+)', r'answer.*?(\d+)', r'结果.*?(\d+)']
    if any(re.search(pattern, solution_str, re.IGNORECASE) for pattern in answer_patterns):
        score += 0.2
    
    # 检查步骤清晰度
    if has_clear_steps(solution_str):
        score += 0.2
    
    return score


def qa_specific_reward(solution_str: str) -> float:
    """问答专用奖励函数"""
    score = 0.0
    
    # 检查回答完整性
    if is_complete_answer(solution_str):
        score += 0.3
    
    # 检查是否提供了解释或细节
    if has_explanations(solution_str):
        score += 0.2
    
    return score


def politeness_reward(solution_str: str) -> float:
    """礼貌用语奖励"""
    score = 0.0
    
    polite_words = [
        "请", "谢谢", "抱歉", "很高兴", "帮助", "please", "thank", "sorry", "glad", "help"
    ]
    
    polite_count = sum(1 for word in polite_words if word in solution_str.lower())
    score += min(polite_count * 0.05, 0.1)  # 最多0.1分
    
    return score


def has_good_structure(text: str) -> bool:
    """检查文本是否有良好的结构"""
    # 检查是否有段落分隔
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        return True
    
    # 检查是否有列表或步骤
    list_patterns = [r'^\d+\.', r'^-', r'^\*', r'首先', r'然后', r'最后']
    if any(re.search(pattern, text, re.MULTILINE) for pattern in list_patterns):
        return True
    
    return False


def has_clear_steps(text: str) -> bool:
    """检查是否有清晰的步骤"""
    step_patterns = [
        r'步骤\s*\d+', r'第\s*\d+\s*步', r'step\s*\d+', 
        r'首先', r'然后', r'接下来', r'最后', r'finally'
    ]
    step_count = sum(1 for pattern in step_patterns if re.search(pattern, text, re.IGNORECASE))
    return step_count >= 2


def is_complete_answer(text: str) -> bool:
    """检查是否是完整的回答"""
    # 检查长度
    if len(text.split()) < 5:
        return False
    
    # 检查是否有结论性词汇
    conclusion_words = ["总之", "因此", "所以", "综上", "conclusion", "therefore", "thus"]
    return any(word in text.lower() for word in conclusion_words)


def has_explanations(text: str) -> bool:
    """检查是否包含解释"""
    explanation_words = [
        "因为", "由于", "原因", "解释", "说明", "because", "since", "reason", "explain"
    ]
    return any(word in text.lower() for word in explanation_words)


def creative_writing_reward(data_source: str, solution_str: str, ground_truth: str, 
                           extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """创意写作专用奖励函数"""
    score = 0.0
    
    # 创意性检查
    if has_creativity(solution_str):
        score += 0.4
    
    # 文学性检查
    if has_literary_quality(solution_str):
        score += 0.3
    
    # 结构完整性
    if has_story_structure(solution_str):
        score += 0.3
    
    return min(score, 1.0)


def has_creativity(text: str) -> bool:
    """检查创意性"""
    # 检查形容词丰富度
    adjective_patterns = [r'\w+的', r'\w+ly', r'beautiful', r'amazing', r'wonderful']
    adj_count = sum(1 for pattern in adjective_patterns 
                   if len(re.findall(pattern, text, re.IGNORECASE)) > 0)
    
    # 检查修辞手法
    rhetoric_patterns = [r'像.*一样', r'如同', r'仿佛', r'like.*as', r'metaphor']
    rhetoric_count = sum(1 for pattern in rhetoric_patterns 
                        if re.search(pattern, text, re.IGNORECASE))
    
    return adj_count >= 3 or rhetoric_count >= 1


def has_literary_quality(text: str) -> bool:
    """检查文学性"""
    # 检查感情色彩
    emotion_words = ["美丽", "悲伤", "快乐", "温暖", "寒冷", "beautiful", "sad", "happy"]
    emotion_count = sum(1 for word in emotion_words if word in text.lower())
    
    return emotion_count >= 2


def has_story_structure(text: str) -> bool:
    """检查故事结构"""
    # 检查是否有开头、发展、结尾
    structure_words = [
        ["从前", "曾经", "在", "once", "long ago"],  # 开头
        ["突然", "然后", "接着", "suddenly", "then"],  # 发展  
        ["最后", "最终", "结果", "finally", "end"]      # 结尾
    ]
    
    structure_found = 0
    for word_group in structure_words:
        if any(word in text.lower() for word in word_group):
            structure_found += 1
    
    return structure_found >= 2


def customer_service_reward(data_source: str, solution_str: str, ground_truth: str,
                           extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """客服对话专用奖励函数"""
    score = 0.0
    
    # 礼貌性 (最重要)
    score += politeness_reward(solution_str) * 3  # 放大礼貌奖励
    
    # 解决方案导向
    if provides_solution(solution_str):
        score += 0.3
    
    # 共情能力
    if shows_empathy(solution_str):
        score += 0.2
    
    return min(score, 1.0)


def provides_solution(text: str) -> bool:
    """检查是否提供解决方案"""
    solution_words = [
        "解决", "处理", "建议", "推荐", "可以", "solve", "fix", "recommend", "suggest"
    ]
    return any(word in text.lower() for word in solution_words)


def shows_empathy(text: str) -> bool:
    """检查是否展现共情"""
    empathy_words = [
        "理解", "明白", "同情", "抱歉", "遗憾", "understand", "sorry", "apologize"
    ]
    return any(word in text.lower() for word in empathy_words)


# 可以根据需要添加更多专用奖励函数...
def compute_score(data_source: str, solution_str: str, ground_truth: str,
                 extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """
    默认的规则奖励函数入口
    
    这是配置 custom_reward_function.name=compute_score 时调用的函数
    """
    return my_custom_reward_function(data_source, solution_str, ground_truth, extra_info, **kwargs)