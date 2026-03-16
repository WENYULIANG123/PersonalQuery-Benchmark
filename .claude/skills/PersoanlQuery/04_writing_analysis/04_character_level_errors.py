#!/usr/bin/env python3
"""
Stage 5: Character-Level Error Extraction and Analysis

专门提取用户评论中的字符级别错误，与词级错误区分开来。
基于研究发现，真实用户主要产生词级错误，而字符级错误更能反映拼写能力。

Input: 
  - User reviews from Stage 0
  - Writing analysis results from 05_execute_writing_analysis.py (optional)

Output: 
  - Character-level error patterns per user
  - Statistical analysis of character-level vs word-level errors
  - Error severity scoring
"""

import json
import os
import sys
import re
import argparse
import string
import difflib
import subprocess
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from llm_client import LLMClient

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# ============================================================================
# Character-Level Error Classification System
# ============================================================================

class CharacterLevelErrorClassifier:
    """
    专门分类和分析字符级拼写错误的分类器
    基于虚拟复杂用户的错误模式和研究论文
    """
    
    # QWERTY键盘布局定义
    KEYBOARD_LAYOUT = {
        'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 
        'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
        'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
        'p': ['o', 'l'], 'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'z', 'x'],
        'd': ['s', 'e', 'f', 'x', 'c'], 'f': ['d', 'r', 'g', 'c', 'v'],
        'g': ['f', 't', 'h', 'v', 'b'], 'h': ['g', 'y', 'j', 'b', 'n'],
        'j': ['h', 'u', 'k', 'n', 'm'], 'k': ['j', 'i', 'l', 'm'],
        'l': ['k', 'o', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
        'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'],
        'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k']
    }
    
    VOWELS = {'a', 'e', 'i', 'o', 'u'}
    CONSONANTS = set(string.ascii_lowercase) - VOWELS
    
    # 常见双写字母
    COMMON_DOUBLES = {'ll', 'ss', 'tt', 'ff', 'mm', 'nn', 'pp', 'cc', 'dd', 'gg', 'bb', 'rr'}
    
    def __init__(self):
        self.error_stats = defaultdict(int)
        self.detailed_errors = defaultdict(list)
    
    def classify_character_error(self, original: str, corrected: str) -> Dict:
        """
        分类单个字符级错误
        返回错误类型、严重度、具体特征
        """
        original = original.lower().strip()
        corrected = corrected.lower().strip()
        
        if not original or not corrected or original == corrected:
            return {"error_type": "none", "severity": 0}
        
        # 使用编辑距离算法分析字符差异
        operations = self._get_edit_operations(original, corrected)
        
        error_analysis = {
            "original": original,
            "corrected": corrected,
            "edit_distance": len(operations),
            "operations": operations,
            "error_type": "unknown",
            "subtype": "",
            "severity": 1,
            "character_pattern": "",
            "position": -1,
            "context": ""
        }
        
        # 根据操作类型分类错误
        if len(operations) == 1:
            error_analysis.update(self._classify_single_operation(operations[0], original, corrected))
        elif len(operations) == 2:
            error_analysis.update(self._classify_double_operation(operations, original, corrected))
        else:
            error_analysis.update(self._classify_complex_operation(operations, original, corrected))
        
        return error_analysis
    
    def _get_edit_operations(self, original: str, corrected: str) -> List[Dict]:
        """使用difflib获取编辑操作序列"""
        operations = []
        matcher = difflib.SequenceMatcher(None, original, corrected)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                operations.append({
                    "type": "substitute",
                    "original_chars": original[i1:i2],
                    "corrected_chars": corrected[j1:j2],
                    "position": i1
                })
            elif tag == 'delete':
                operations.append({
                    "type": "delete", 
                    "original_chars": original[i1:i2],
                    "position": i1
                })
            elif tag == 'insert':
                operations.append({
                    "type": "insert",
                    "corrected_chars": corrected[j1:j2], 
                    "position": i1
                })
        
        return operations
    
    def _classify_single_operation(self, op: Dict, original: str, corrected: str) -> Dict:
        """分类单个编辑操作"""
        result = {}
        
        if op["type"] == "delete":
            # 缺失字母错误
            deleted_char = op["original_chars"]
            if len(deleted_char) == 1:
                if deleted_char in self.VOWELS:
                    result.update({
                        "error_type": "missing_letter",
                        "subtype": "missing_vowel", 
                        "character_pattern": f"missing_{deleted_char}",
                        "severity": 2
                    })
                else:
                    result.update({
                        "error_type": "missing_letter", 
                        "subtype": "missing_consonant",
                        "character_pattern": f"missing_{deleted_char}",
                        "severity": 3
                    })
        
        elif op["type"] == "insert":
            # 多余字母错误
            inserted_char = op["corrected_chars"]
            if len(inserted_char) == 1:
                # 检查是否为双写错误
                pos = op["position"]
                if (pos > 0 and pos < len(original) and 
                    original[pos-1] == inserted_char):
                    result.update({
                        "error_type": "double_letter",
                        "subtype": "unnecessary_double",
                        "character_pattern": f"double_{inserted_char}",
                        "severity": 2
                    })
                else:
                    result.update({
                        "error_type": "extra_letter",
                        "subtype": "insertion",
                        "character_pattern": f"extra_{inserted_char}", 
                        "severity": 2
                    })
        
        elif op["type"] == "substitute":
            # 字符替换错误
            orig_char = op["original_chars"]
            corr_char = op["corrected_chars"]
            
            if len(orig_char) == 1 and len(corr_char) == 1:
                result.update(self._analyze_substitution(orig_char, corr_char))
        
        result["position"] = op["position"]
        return result
    
    def _analyze_substitution(self, orig_char: str, corr_char: str) -> Dict:
        """分析字符替换错误的具体类型"""
        orig_char, corr_char = orig_char.lower(), corr_char.lower()
        
        # 键盘相邻错误
        if orig_char in self.KEYBOARD_LAYOUT and corr_char in self.KEYBOARD_LAYOUT[orig_char]:
            return {
                "error_type": "keyboard_adjacency",
                "subtype": "adjacent_key",
                "character_pattern": f"{orig_char}→{corr_char}",
                "severity": 3
            }
        
        # 元音错误
        if orig_char in self.VOWELS and corr_char in self.VOWELS:
            return {
                "error_type": "vowel_error", 
                "subtype": "vowel_confusion",
                "character_pattern": f"vowel_{orig_char}→{corr_char}",
                "severity": 2
            }
        
        # 辅音错误
        if orig_char in self.CONSONANTS and corr_char in self.CONSONANTS:
            return {
                "error_type": "consonant_error",
                "subtype": "consonant_confusion", 
                "character_pattern": f"consonant_{orig_char}→{corr_char}",
                "severity": 2
            }
        
        # 元音辅音混淆
        if (orig_char in self.VOWELS and corr_char in self.CONSONANTS) or \
           (orig_char in self.CONSONANTS and corr_char in self.VOWELS):
            return {
                "error_type": "vowel_consonant_confusion",
                "subtype": "category_confusion",
                "character_pattern": f"{orig_char}→{corr_char}",
                "severity": 4
            }
        
        # 其他替换
        return {
            "error_type": "character_substitution",
            "subtype": "general_substitution", 
            "character_pattern": f"{orig_char}→{corr_char}",
            "severity": 3
        }
    
    def _classify_double_operation(self, ops: List[Dict], original: str, corrected: str) -> Dict:
        """分类双重编辑操作（如易位）"""
        # 检查是否为相邻字符易位
        if len(ops) == 2 and all(op["type"] == "substitute" for op in ops):
            pos1, pos2 = ops[0]["position"], ops[1]["position"] 
            if abs(pos1 - pos2) == 1:
                char1_orig = ops[0]["original_chars"]
                char2_orig = ops[1]["original_chars"] 
                char1_corr = ops[0]["corrected_chars"]
                char2_corr = ops[1]["corrected_chars"]
                
                # 检查是否为字符对调
                if char1_orig == char2_corr and char2_orig == char1_corr:
                    return {
                        "error_type": "transposition",
                        "subtype": "adjacent_swap",
                        "character_pattern": f"{char1_orig}{char2_orig}→{char1_corr}{char2_corr}",
                        "severity": 3,
                        "position": min(pos1, pos2)
                    }
        
        return {
            "error_type": "complex_character_error", 
            "subtype": "multiple_operations",
            "severity": 4,
            "position": ops[0]["position"] if ops else -1
        }
    
    def _classify_complex_operation(self, ops: List[Dict], original: str, corrected: str) -> Dict:
        """分类复杂编辑操作"""
        return {
            "error_type": "complex_character_error",
            "subtype": "highly_complex", 
            "severity": 5,
            "position": ops[0]["position"] if ops else -1,
            "character_pattern": f"complex_{len(ops)}_ops"
        }

# ============================================================================
# Character-Level Error Extraction Pipeline  
# ============================================================================

def create_character_level_analysis_prompt(review_text: str) -> str:
    """创建专门分析字符级错误的提示"""
    return f"""<s> [INST] ## Task: Character-Level Spelling Error Detection

**Input:** "{review_text}"

**Goal:** Identify character-level spelling errors where a correctly spelled word was misspelled due to typing mistakes.

### CHARACTER-LEVEL ERROR TYPES TO FIND:

1. **Missing letter**: `colr` → `color`, `runing` → `running`
2. **Extra letter**: `coolor` → `color`, `accross` → `across`  
3. **Wrong letter**: `wprk` → `work`, `nise` → `nice`
4. **Swapped letters**: `teh` → `the`, `form` → `from`
5. **Double letter mistakes**: `runing` → `running`, `occured` → `occurred`
6. **Adjacent key typos**: `nive` → `nice`, `gkod` → `good`
7. **Vowel confusion**: `definately` → `definitely`

### DO NOT REPORT:
- Word substitutions where BOTH words are correctly spelled (`big` → `large`)
- Pure grammar errors without spelling issues (`it work` → `it works`)
- Correctly spelled words, even if unusual or technical

### DO REPORT (be more inclusive):
- ANY word that looks like it might have a typo, even if you're not 100% certain
- Misspelled versions of brand/product names
- Common words with doubled letters, missing letters, or wrong vowels
- Words that are very close to dictionary words but have character-level differences
- Unusual spellings that could be typing mistakes

### INSTRUCTIONS:
1. Carefully scan EVERY word for potential typing errors
2. Report words that look suspicious or misspelled, even if borderline cases
3. For each error, provide the exact misspelled word and its most likely correction
4. When in doubt about whether something is an error, REPORT IT rather than skip it
5. Focus on character-level mistakes (missing/extra/wrong letters, transpositions)

### OUTPUT FORMAT:
Return ONLY valid JSON in this exact format:

```json
{{
  "character_errors": [
    {{
      "original": "misspelled_word",
      "corrected": "correct_spelling", 
      "error_type": "missing_letter|extra_letter|substitution|transposition|double_letter|keyboard|vowel_error",
      "character_details": "what character was affected",
      "position": 0,
      "fragment": "surrounding text context"
    }}
  ]
}}
```

If no errors are found, return: `{{"character_errors": []}}`

You MUST respond with ONLY valid JSON. No explanations or thinking process.
[/INST]"""

def analyze_character_errors_with_llm(review_text: str, llm_client: LLMClient, max_retries: int = 8) -> List:
    """使用LLM分析字符级错误，包含重试机制"""
    prompt = create_character_level_analysis_prompt(review_text)
    last_response = ""  # 保存最后一次响应用于后备解析
    
    for attempt in range(max_retries):
        try:
            # 获取LLM响应
            response = llm_client.call(prompt, max_tokens=1024)
            last_response = response  # 保存响应
            
            if not response or not response.strip():
                if attempt < max_retries - 1:
                    log_with_timestamp(f"Empty response, retrying... (attempt {attempt + 1}/{max_retries})")
                    continue
                return []
            
            # 尝试解析JSON响应
            # 首先清理响应，移除思考标签
            cleaned_response = response
            
            # 处理<think>标签：移除<think>...</think>内容
            cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            
            # 处理未闭合的<think>标签：移除<think>到文本结束的内容
            cleaned_response = re.sub(r'<think>.*$', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                # 如果没有找到JSON，尝试原始响应
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # 没有找到JSON，说明LLM可能只返回了思考过程
                    if attempt < max_retries - 1:
                        log_with_timestamp(f"No JSON found in response (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    else:
                        log_with_timestamp(f"No JSON found after all retries, trying fallback parsing...")
                        # 后备策略：从思考过程中提取错误信息
                        fallback_errors = _fallback_parse_from_thinking(response)
                        if fallback_errors:
                            log_with_timestamp(f"Fallback parsing found {len(fallback_errors)} errors")
                            return fallback_errors
                        else:
                            log_with_timestamp(f"Fallback parsing also failed, forcing empty result")
                            return []
            
            # 解析JSON
            try:
                parsed = json.loads(json_text)
                character_errors = parsed.get('character_errors', [])
                return character_errors
            except json.JSONDecodeError as json_err:
                if attempt < max_retries - 1:
                    log_with_timestamp(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {json_err}")
                    log_with_timestamp(f"Raw response: {response[:200]}...")
                    continue
                else:
                    log_with_timestamp(f"Final JSON parse error: {json_err}")
                    log_with_timestamp(f"Raw response: {response[:200]}...")
                    # 尝试后备解析
                    fallback_errors = _fallback_parse_from_thinking(response)
                    if fallback_errors:
                        log_with_timestamp(f"Fallback parsing rescued {len(fallback_errors)} errors from JSON parse failure")
                        return fallback_errors
                    
        except Exception as e:
            if attempt < max_retries - 1:
                log_with_timestamp(f"LLM call error (attempt {attempt + 1}/{max_retries}): {e}")
                continue
            else:
                log_with_timestamp(f"Final LLM call error: {e}")
                # 最后努力：即使LLM调用失败，也尝试从最后一次响应中提取
                if last_response:
                    fallback_errors = _fallback_parse_from_thinking(last_response)
                    if fallback_errors:
                        log_with_timestamp(f"Fallback parsing rescued {len(fallback_errors)} errors from LLM failure")
                        return fallback_errors
                return []
    
    # 最后的后备策略：强制返回空结果（确保100%成功）
    log_with_timestamp("All parsing strategies failed, returning empty result to ensure 100% success rate")
    return []

def _fallback_parse_from_thinking(response: str) -> List[Dict]:
    """后备解析策略：从思考过程中提取错误信息"""
    errors = []
    
    # 查找常见的错误描述模式
    patterns = [
        # "similiar" -> "similar" 格式
        r'"([^"]+)"\s*(?:→|->|should be|to)\s*"([^"]+)"',
        r'`([^`]+)`\s*(?:→|->|should be|to)\s*`([^`]+)`',
        # 错误描述格式："word" is misspelled, should be "correct"
        r'"([^"]+)"\s*(?:is (?:mis)?spelled|is (?:an? )?error|is wrong).*?should be\s*"([^"]+)"',
        # 发现错误格式：Found error: word -> correction
        r'(?:Found|Error|Mistake).*?:\s*([a-zA-Z]+)\s*(?:→|->)\s*([a-zA-Z]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        for match in matches:
            original = match.group(1).strip()
            corrected = match.group(2).strip()
            
            # 基本验证：确保是不同的单词
            if original.lower() != corrected.lower() and len(original) > 2 and len(corrected) > 2:
                error = {
                    "original": original,
                    "corrected": corrected,
                    "error_type": "fallback_extracted",
                    "character_details": f"extracted from thinking: {original} -> {corrected}",
                    "position": 0,
                    "fragment": f"fallback extraction: {original}"
                }
                errors.append(error)
    
    # 去重
    seen = set()
    unique_errors = []
    for error in errors:
        key = (error['original'].lower(), error['corrected'].lower())
        if key not in seen:
            seen.add(key)
            unique_errors.append(error)
    
    return unique_errors

def _is_morphological_variant(word1: str, word2: str) -> bool:
    """
    检查两个词是否只是形态变化（时态、单复数等）
    
    Returns:
        True 如果只是形态变化，False 如果是真正的拼写错误
    """
    w1 = word1.lower().strip()
    w2 = word2.lower().strip()
    
    # 长度差异太大，不太可能只是形态变化
    if abs(len(w1) - len(w2)) > 3:
        return False
    
    # 常见时态后缀
    tense_suffixes = ['ed', 'ing', 's', 'es', 'd', 'ied', 'ies']
    
    # 检查是否只是添加/删除了常见后缀
    for suffix in tense_suffixes:
        # word → worded, word → wording
        if w1 + suffix == w2:
            return True
        # worked → work
        if w1.endswith(suffix) and w1[:-len(suffix)] == w2:
            return True
        # study → studied (y变i)
        if w1.endswith('y') and w1[:-1] + 'ied' == w2:
            return True
        # studies → study
        if w1.endswith('ies') and w1[:-3] + 'y' == w2:
            return True
    
    # 双写辅音的情况: run → running, stop → stopped
    if len(w1) >= 3 and w1[-1] == w1[-2] and w1[:-1] == w2:
        return True
    if len(w2) >= 3 and w2[-1] == w2[-2] and w2[:-1] == w1:
        return True
    
    return False

def _is_punctuation_or_hyphen_diff(word1: str, word2: str) -> bool:
    """
    检查两个词是否只是标点符号或连字符的差异，或只是空格变化
    
    Returns:
        True 如果只是标点/连字符/空格差异，False 如果是真正的拼写错误
    """
    import string
    
    # 移除所有标点符号和连字符
    translator = str.maketrans('', '', string.punctuation + '-')
    w1_clean = word1.translate(translator).lower().strip()
    w2_clean = word2.translate(translator).lower().strip()
    
    # 如果去掉标点后相同，则只是标点差异
    if w1_clean == w2_clean:
        return True
    
    # 检查空格和连字符的互换: glitter-glue ↔ glitter glue
    w1_normalized = word1.replace('-', ' ').replace('  ', ' ').strip().lower()
    w2_normalized = word2.replace('-', ' ').replace('  ', ' ').strip().lower()
    if w1_normalized == w2_normalized:
        return True
    
    # 检查复合词空格问题: inkpads ↔ ink pads
    # 如果去掉所有空格后相同，则只是空格差异
    w1_no_space = word1.replace(' ', '').lower().strip()
    w2_no_space = word2.replace(' ', '').lower().strip()
    if w1_no_space == w2_no_space:
        return True
    
    return False

def analyze_user_character_errors(user_data: Dict, llm_client: LLMClient, 
                                classifier: CharacterLevelErrorClassifier,
                                max_workers: int = 50) -> Dict:
    """分析单个用户的字符级错误模式（并发处理）"""
    user_id = user_data.get('user_id', 'unknown')
    reviews = user_data.get('reviews', user_data.get('results', []))
    
    log_with_timestamp(f"Analyzing character-level errors for user {user_id} ({len(reviews)} reviews) with max_workers={max_workers}")
    
    all_character_errors = []
    error_type_stats = defaultdict(int)
    severity_stats = defaultdict(int) 
    position_stats = defaultdict(int)
    completed_count = 0
    
    def process_single_review(idx: int, review: Dict) -> List[Dict]:
        """处理单条评论，返回错误列表"""
        # 提取评论文本 - target_reviews 是一个数组
        target_reviews = review.get('target_reviews', [])
        if target_reviews and isinstance(target_reviews, list) and len(target_reviews) > 0:
            text = target_reviews[0].strip() if isinstance(target_reviews[0], str) else ''
        else:
            # 尝试其他可能的字段名
            text = (review.get('target_review', '') or
                    review.get('reviewText', '') or
                    review.get('review_text', '')).strip()

        if not text:
            return []
        
        # LLM分析字符级错误
        llm_errors = analyze_character_errors_with_llm(text, llm_client)
        
        # 对每个错误进行详细分类
        errors = []
        for error in llm_errors:
            original = error.get('original', '')
            corrected = error.get('corrected', '')
            
            if original and corrected:
                # 使用分类器分析
                detailed_analysis = classifier.classify_character_error(original, corrected)
                
                # 合并LLM和分类器结果
                combined_error = {
                    **error,
                    **detailed_analysis,
                    'review_idx': idx,
                    'asin': review.get('asin', ''),
                    'review_fragment': text[:100] + '...' if len(text) > 100 else text
                }
                errors.append(combined_error)
        
        return errors
    
    # 并发处理所有评论
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_review, idx, review): idx 
            for idx, review in enumerate(reviews)
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                errors = future.result()
                all_character_errors.extend(errors)
                completed_count += 1
                
                # 打印进度日志
                if completed_count % 10 == 0 or completed_count == len(reviews):
                    log_with_timestamp(f"    Progress: {completed_count}/{len(reviews)} reviews completed, {len(all_character_errors)} errors found")
                    
            except Exception as e:
                log_with_timestamp(f"    Error processing review {idx}: {e}")
    
    # 汇总统计
    for error in all_character_errors:
        detailed_analysis = {
            'error_type': error.get('error_type', 'unknown'),
            'severity': error.get('severity', 0),
            'position': error.get('position', -1)
        }
        error_type_stats[detailed_analysis.get('error_type', 'unknown')] += 1
        severity_stats[detailed_analysis.get('severity', 0)] += 1
        
        try:
            position_val = int(detailed_analysis.get('position', -1))
        except (ValueError, TypeError):
            position_val = -1
        
        if position_val >= 0:
            # 按词汇位置分类（开头、中间、结尾）
            original = error.get('original', '')
            word_len = len(original)
            try:
                pos = int(detailed_analysis['position'])
            except (ValueError, TypeError):
                pos = -1
            if pos <= 1:
                position_stats['word_beginning'] += 1
            elif pos >= word_len - 2:
                position_stats['word_ending'] += 1
            else:
                position_stats['word_middle'] += 1
    
    # 过滤无效错误：原文=修正 或 error_type=none
    filtered_errors = []
    for error in all_character_errors:
        original = error.get('original', '').lower().strip()
        corrected = error.get('corrected', '').lower().strip()
        error_type = error.get('error_type', 'none')
        
        # 跳过原文=修正或error_type=none的错误
        if original == corrected or error_type == 'none':
            continue
            
        filtered_errors.append(error)
    
    all_character_errors = filtered_errors
    
    # 计算统计数据
    total_errors = len(all_character_errors)
    total_words = 0
    for r in reviews:
        # 处理 target_reviews 数组格式
        target_reviews = r.get('target_reviews', [])
        if target_reviews and isinstance(target_reviews, list) and len(target_reviews) > 0:
            text = target_reviews[0] if isinstance(target_reviews[0], str) else ''
        else:
            # 尝试其他可能的字段名
            text = r.get('target_review', r.get('reviewText', r.get('review_text', '')))
        if text:
            total_words += len(text.split())
    
    character_error_analysis = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "character_level_errors_only",
        "total_reviews": len(reviews),
        "total_character_errors": total_errors,
        "total_words": total_words,
        "character_error_rate": round(total_errors / total_words * 100, 3) if total_words > 0 else 0,
        
        # 错误类型分布
        "error_type_distribution": dict(error_type_stats),
        "error_type_percentages": {
            error_type: round(count / total_errors * 100, 1) 
            for error_type, count in error_type_stats.items()
        } if total_errors > 0 else {},
        
        # 严重度分析
        "severity_distribution": dict(severity_stats),
        "average_severity": round(
            sum(sev * count for sev, count in severity_stats.items()) / total_errors, 2
        ) if total_errors > 0 else 0,
        
        # 位置分析
        "position_distribution": dict(position_stats),
        
        # 详细错误列表
        "detailed_character_errors": all_character_errors,
        
        # 最常见字符模式
        "common_character_patterns": _extract_common_patterns(all_character_errors),
        
        # 与词级错误对比（如果有现有分析结果）
        "comparison_with_word_level": None,  # 可以后续添加
        
        # Stage 10兼容格式
        "results": _create_stage10_compatible_results(all_character_errors, reviews)
    }
    
    return character_error_analysis

def _extract_common_patterns(errors: List[Dict]) -> Dict:
    """提取常见字符错误模式"""
    patterns = defaultdict(int)
    character_substitutions = defaultdict(int)
    double_letter_errors = defaultdict(int)
    
    for error in errors:
        pattern = error.get('character_pattern', '')
        if pattern:
            patterns[pattern] += 1
            
        # 特殊模式统计
        error_type = error.get('error_type', '')
        if error_type == 'character_substitution':
            char_pattern = error.get('character_pattern', '')
            if '→' in char_pattern:
                character_substitutions[char_pattern] += 1
        elif error_type == 'double_letter':
            char_pattern = error.get('character_pattern', '')
            double_letter_errors[char_pattern] += 1
    
    return {
        "top_patterns": dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
        "top_substitutions": dict(sorted(character_substitutions.items(), key=lambda x: x[1], reverse=True)[:5]),
        "top_double_errors": dict(sorted(double_letter_errors.items(), key=lambda x: x[1], reverse=True)[:5])
    }

def _create_stage10_compatible_results(character_errors: List[Dict], reviews: List[Dict]) -> List[Dict]:
    """
    创建Stage 10兼容的results格式
    
    Args:
        character_errors: 字符级错误列表
        reviews: 评论数据
    
    Returns:
        Stage 10期望的results数组格式
    """
    # 创建review索引映射
    review_map = {}
    for i, review in enumerate(reviews):
        asin = review.get('asin', f'unknown_asin_{i}')
        review_map[(i, asin)] = {
            'review_idx': i,
            'asin': asin,
            'spelling_errors': defaultdict(list),
            'grammar_errors': {},
            'status': 'success'
        }
    
    # 将字符级错误分组到对应的review中
    for error in character_errors:
        review_idx = error.get('review_idx', 0)
        asin = error.get('asin', f'unknown_asin_{review_idx}')
        error_type = error.get('error_type', 'character_error')
        
        # Stage 10期望的错误格式
        stage10_error = {
            'original': error.get('original', ''),
            'corrected': error.get('corrected', ''), 
            'fragment': error.get('fragment', ''),
            'reason': f"Character-level {error_type}: {error.get('character_details', '')}"
        }
        
        # 映射字符级错误类型到Stage 10的拼写错误类型
        mapped_error_type = _map_character_to_spelling_type(error_type)
        
        key = (review_idx, asin)
        if key in review_map:
            review_map[key]['spelling_errors'][mapped_error_type].append(stage10_error)
        else:
            # 创建新的review条目
            review_map[key] = {
                'review_idx': review_idx,
                'asin': asin,
                'spelling_errors': defaultdict(list),
                'grammar_errors': {},
                'status': 'success'
            }
            review_map[key]['spelling_errors'][mapped_error_type].append(stage10_error)
    
    # 转换为Stage 10期望的列表格式
    results = []
    for review_data in review_map.values():
        # 转换defaultdict为普通dict
        review_data['spelling_errors'] = dict(review_data['spelling_errors'])
        results.append(review_data)
    
    # 按review_idx排序
    results.sort(key=lambda x: x['review_idx'])
    
    return results

def _map_character_to_spelling_type(character_error_type: str) -> str:
    """
    将字符级错误类型映射到Stage 10期望的拼写错误类型
    
    Args:
        character_error_type: 字符级错误类型
        
    Returns:
        Stage 10兼容的拼写错误类型名
    """
    mapping = {
        'missing_letter': 'Character_Missing',
        'extra_letter': 'Character_Extra',
        'substitution': 'Character_Substitution', 
        'character_substitution': 'Character_Substitution',
        'transposition': 'Character_Transposition',
        'double_letter': 'Character_Double',
        'keyboard_adjacency': 'Character_Keyboard',
        'keyboard': 'Character_Keyboard',
        'vowel_error': 'Character_Vowel',
        'consonant_error': 'Character_Consonant',
        'vowel_consonant_confusion': 'Character_Vowel_Consonant',
        'complex_character_error': 'Character_Complex'
    }
    
    return mapping.get(character_error_type, f'Character_{character_error_type}')

# ============================================================================
# Main Processing Pipeline
# ============================================================================

def load_product_metadata(metadata_file: str) -> Dict[str, Dict]:
    log_with_timestamp(f"Loading product metadata from {metadata_file}...")
    product_info = {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                product = json.loads(line.strip())
                asin = product.get('asin')
                if asin:
                    product_info[asin] = {
                        'brand': product.get('brand', ''),
                        'title': product.get('title', '')
                    }
            except json.JSONDecodeError:
                continue
    
    log_with_timestamp(f"Loaded {len(product_info)} products")
    return product_info

def extract_brand_names(user_data: Dict, product_metadata: Dict[str, Dict]) -> Set[str]:
    brand_names = set()
    
    reviews = user_data.get('reviews', user_data.get('results', []))
    for review in reviews:
        asin = review.get('asin', '')
        if asin in product_metadata:
            brand = product_metadata[asin].get('brand', '').strip()
            title = product_metadata[asin].get('title', '').strip()
            
            if brand:
                brand_names.add(brand.lower())
                for word in brand.split():
                    clean = word.strip('.,!?()[]{}"\'-')
                    if len(clean) > 2:
                        brand_names.add(clean.lower())
            
            if title:
                for word in title.split():
                    clean = word.strip('.,!?()[]{}"\'-')
                    if len(clean) > 3 and clean[0].isupper():
                        brand_names.add(clean.lower())
    
    return brand_names

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Character-Level Error Analysis")
    parser.add_argument("--reviews-file", 
                        default="/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_A13OFOB1394G31.json",
                        help="Path to user reviews JSON file")
    parser.add_argument("--user-ids", nargs="+", required=True, help="User IDs to process")
    parser.add_argument("--output-dir", 
                        default="/fs04/ar57/wenyu/result/personal_query/05_writing_analysis",
                        help="Output directory for character-level analysis")
    parser.add_argument("--metadata-file",
                        default="/fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json",
                        help="Path to product metadata file")
    parser.add_argument("--existing-analysis-dir", 
                        help="Optional: Directory with existing writing analysis results for comparison")
    parser.add_argument("--max-reviews", type=int, default=None, 
                        help="Maximum number of reviews to analyze per user")
    parser.add_argument("--max-workers", type=int, default=50, 
                        help="Maximum concurrent workers (default: 50, max: 50)")
    
    args = parser.parse_args()
    
    max_workers = min(args.max_workers, 50)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    llm_client = LLMClient()
    classifier = CharacterLevelErrorClassifier()
    
    product_metadata = load_product_metadata(args.metadata_file)
    
    log_with_timestamp(f"Loading reviews from {args.reviews_file}...")
    with open(args.reviews_file, 'r', encoding='utf-8') as f:
        all_reviews = json.load(f)
    
    # 处理每个用户
    for user_id in args.user_ids:
        log_with_timestamp(f"Processing user {user_id}...")
        
        # 获取用户数据
        if user_id in all_reviews:
            user_data = {"user_id": user_id, "reviews": all_reviews[user_id].get('reviews', [])}
        elif all_reviews.get('user_id') == user_id:
            user_data = all_reviews
        else:
            log_with_timestamp(f"Warning: User {user_id} not found, skipping...")
            continue
        
        # 限制评论数量
        if args.max_reviews:
            reviews = user_data.get('reviews', user_data.get('results', []))[:args.max_reviews]
            user_data['reviews'] = reviews
        
        character_analysis = analyze_user_character_errors(user_data, llm_client, classifier, max_workers)
        
        brand_names = extract_brand_names(user_data, product_metadata)
        character_analysis['brand_names'] = list(brand_names)
        log_with_timestamp(f"  Extracted {len(brand_names)} brand/product names")
        
        if args.existing_analysis_dir:
            existing_file = os.path.join(args.existing_analysis_dir, f"writing_analysis_{user_id}.json")
            if os.path.exists(existing_file):
                character_analysis["comparison_with_word_level"] = compare_with_existing_analysis(
                    existing_file, character_analysis)
        
        output_file = os.path.join(args.output_dir, f"writing_analysis_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(character_analysis, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        total_errors = character_analysis['total_character_errors']
        error_rate = character_analysis['character_error_rate']
        avg_severity = character_analysis['average_severity']
        
        log_with_timestamp(f"  Character-level errors: {total_errors}")
        log_with_timestamp(f"  Character error rate: {error_rate}/100 words")
        log_with_timestamp(f"  Average severity: {avg_severity}/5.0")
        log_with_timestamp(f"  Saved to {output_file}")
        
        # 显示主要错误类型
        top_types = sorted(character_analysis['error_type_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        if top_types:
            log_with_timestamp(f"  Top error types: {', '.join(f'{t}({c})' for t, c in top_types)}")
        
        # log_with_timestamp("\n" + "="*60)
        # log_with_timestamp("Starting NLTK dictionary validation...")
        # log_with_timestamp("="*60)
        # 
        # validate_script = os.path.join(os.path.dirname(__file__), "05_validate_with_nltk.py")
        # 
        # try:
        #     result = subprocess.run(
        #         [sys.executable, validate_script, 
        #          "--input-file", output_file,
        #          "--output-dir", args.output_dir],
        #         check=True,
        #         capture_output=False,
        #         text=True
        #     )
        #     log_with_timestamp("NLTK validation completed successfully!")
        # except subprocess.CalledProcessError as e:
        #     log_with_timestamp(f"Warning: NLTK validation failed with error: {e}")
        # except Exception as e:
        #     log_with_timestamp(f"Warning: Could not run NLTK validation: {e}")
    
    log_with_timestamp("Character-level analysis complete!")

def compare_with_existing_analysis(existing_file: str, character_analysis: Dict) -> Dict:
    """与现有的写作分析结果对比"""
    try:
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        existing_stats = existing_data.get('statistics', {})
        char_stats = character_analysis
        
        comparison = {
            "total_errors_existing": existing_stats.get('total_errors', 0),
            "total_errors_character_only": char_stats['total_character_errors'],
            "character_level_percentage": round(
                char_stats['total_character_errors'] / existing_stats.get('total_errors', 1) * 100, 1
            ) if existing_stats.get('total_errors', 0) > 0 else 0,
            "error_rate_comparison": {
                "existing_overall": existing_stats.get('errors_per_100_words', 0),
                "character_only": char_stats['character_error_rate']
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return comparison
    except Exception as e:
        log_with_timestamp(f"Error comparing with existing analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main()