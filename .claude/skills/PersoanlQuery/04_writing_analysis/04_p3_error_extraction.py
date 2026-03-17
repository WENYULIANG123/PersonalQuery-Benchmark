#!/usr/bin/env python3
"""
Stage 4B: P3最优模板错误提取

根据MTSummit 2025论文《Exploring the Feasibility of Multilingual Grammatical Error 
Correction with a Single LLM up to 9B parameters》的发现，使用P3最优prompt模板进行
多语言语法错误提取。

P3模板（最优）:
"Edit the following text for spelling and grammar mistakes, make minimal changes, and 
return only the corrected text. If the text is already correct, return it without any 
explanations:"

论文结论: P3在32/36评估场景中表现最优，F1分数提升 +176~283%

Input: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json

Output: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/p3_analysis_{user_id}.json

Usage:
  # Process specific user with P3 template
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID
  
  # Process with custom output directory
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID \\
    --output-dir /path/to/output
    
  # Process with max reviews limit
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID \\
    --max-reviews 100
"""

import json
import os
import sys
import argparse
import re
import difflib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from llm_client import LLMClient


def log_with_timestamp(message: str):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================================
# P3最优提示模板定义
# ============================================================================

class P3ErrorExtractor:
    """
    使用P3最优模板进行错误提取
    
    P3特点：
    1. 明确指导：make minimal changes
    2. 处理正确文本：If the text is already correct, return it without any explanations
    3. 输出规范：return only the corrected text
    
    论文数据：P3相比P1性能提升 +23~283%，特别是F1分数（保持正确文本能力）
    """
    
    # P3最优模板（根据论文4.2节）
    P3_TEMPLATE = """Edit the following text for spelling and grammar mistakes, make minimal changes, and return only the corrected text. If the text is already correct, return it without any explanations:."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.error_stats = defaultdict(int)
    
    def create_p3_prompt(self, review_text: str) -> str:
        """
        创建P3最优提示
        
        Args:
            review_text: 要处理的评论文本
            
        Returns:
            完整的prompt
        """
        return f"""<s>[INST] {self.P3_TEMPLATE}

"{review_text}"

Please return ONLY the corrected text. If no corrections are needed, return the original text exactly as it is.
[/INST]"""
    
    def extract_errors(self, original_text: str, max_retries: int = 5) -> Dict:
        """
        使用P3模板提取错误
        
        Args:
            original_text: 原始文本
            max_retries: 最大重试次数
            
        Returns:
            错误提取结果字典
        """
        if not original_text or not original_text.strip():
            return {
                "original": original_text,
                "corrected": original_text,
                "has_errors": False,
                "error_count": 0,
                "extraction_status": "empty_input"
            }
        
        prompt = self.create_p3_prompt(original_text)
        last_response = ""
        
        for attempt in range(max_retries):
            try:
                # 调用LLM获取纠正后的文本
                response = self.llm_client.call(prompt, max_tokens=512)
                last_response = response
                
                if not response or not response.strip():
                    if attempt < max_retries - 1:
                        log_with_timestamp(f"  Warning: Empty response, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    return {
                        "original": original_text,
                        "corrected": original_text,
                        "has_errors": False,
                        "extraction_status": "empty_response_after_retries"
                    }
                
                # 清理响应（去除思考过程标签）
                cleaned_response = self._clean_response(response)
                
                # 比较原文和纠正文本
                original_clean = original_text.strip()
                corrected_clean = cleaned_response.strip()
                
                has_errors = original_clean != corrected_clean
                
                # 计算错误数量（粗略估计）
                error_count = self._estimate_error_count(original_clean, corrected_clean)
                
                return {
                    "original": original_text,
                    "corrected": cleaned_response,
                    "has_errors": has_errors,
                    "error_count": error_count,
                    "extraction_status": "success",
                    "original_length": len(original_clean.split()),
                    "corrected_length": len(corrected_clean.split())
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    log_with_timestamp(f"  Error: {e} (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    return {
                        "original": original_text,
                        "corrected": original_text,
                        "has_errors": False,
                        "extraction_status": f"error_after_retries: {str(e)}"
                    }
        
        return {
            "original": original_text,
            "corrected": original_text,
            "has_errors": False,
            "extraction_status": "max_retries_exceeded"
        }
    
    def _clean_response(self, response: str) -> str:
        """
        清理LLM响应
        
        移除思考过程和多余的文本
        """
        # 移除 <think> 标签
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        # 移除未闭合的 <think> 标签
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # 移除多余的 [INST] 和相关标签
        cleaned = re.sub(r'\[/INST\]', '', cleaned)
        cleaned = re.sub(r'<s>\[INST\].*?\[/INST\]', '', cleaned, flags=re.DOTALL)
        
        return cleaned.strip()
    
    def _estimate_error_count(self, original: str, corrected: str) -> int:
        """
        计算修正前后的实际改动数量（基于字符级编辑距离）
        
        使用SequenceMatcher计算连续改动块的数量，这比简单词级差异更准确。
        每个改动块代表一个编辑操作（删除、插入、替换）。
        阈值为10字符/错误，允许小的拼写错误和标点符号调整不算作单独错误。
        """
        if original == corrected:
            return 0
        
        matcher = difflib.SequenceMatcher(None, original, corrected)
        matching_blocks = matcher.get_matching_blocks()
        total_chars = len(original)
        matched_chars = sum(block.size for block in matching_blocks)
        changed_chars = total_chars - matched_chars
        
        chars_per_error = 10
        error_count = max(1, changed_chars // chars_per_error + (1 if changed_chars % chars_per_error > 0 else 0))
        
        return error_count if changed_chars > 0 else 0


# ============================================================================
# 批量处理函数
# ============================================================================

def process_user_reviews(
    user_data: Dict,
    output_dir: str,
    llm_client: LLMClient,
    max_workers: int = 20
) -> Dict:
    """
    处理单个用户的所有评论
    
    Args:
        user_data: 用户数据字典
        output_dir: 输出目录
        llm_client: LLM客户端
        max_workers: 并发工作线程数
        
    Returns:
        分析结果
    """
    user_id = user_data.get('user_id', 'unknown')
    reviews = user_data.get('reviews', user_data.get('results', []))
    
    log_with_timestamp(f"Processing user {user_id}: {len(reviews)} reviews")
    
    extractor = P3ErrorExtractor(llm_client)
    all_results = []
    error_stats = defaultdict(int)
    
    def process_single_review(idx: int, review: Dict) -> Dict:
        """处理单条评论"""
        # 提取评论文本
        target_reviews = review.get('target_reviews', [])
        if target_reviews and isinstance(target_reviews, list) and len(target_reviews) > 0:
            text = target_reviews[0].strip() if isinstance(target_reviews[0], str) else ''
        else:
            text = (review.get('target_review', '') or
                    review.get('reviewText', '') or
                    review.get('review_text', '')).strip()
        
        if not text:
            return None
        
        # 使用P3模板提取错误
        result = extractor.extract_errors(text)
        result['review_idx'] = idx
        result['asin'] = review.get('asin', '')
        result['timestamp'] = review.get('reviewTime', '')
        
        return result
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_review, idx, review): idx
            for idx, review in enumerate(reviews)
        }
        
        completed_count = 0
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    # 统计
                    status = result.get('extraction_status', 'unknown')
                    error_stats[status] += 1
                    
                    if result.get('has_errors'):
                        error_stats['has_errors'] += 1
                    else:
                        error_stats['no_errors'] += 1
                
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(reviews):
                    log_with_timestamp(f"  Progress: {completed_count}/{len(reviews)} reviews processed")
                    
            except Exception as e:
                log_with_timestamp(f"  Error processing review: {e}")
    
    # 计算统计信息
    total_reviews = len(all_results)
    reviews_with_errors = error_stats.get('has_errors', 0)
    reviews_without_errors = error_stats.get('no_errors', 0)
    
    total_words = sum(r.get('original_length', 0) for r in all_results if isinstance(r, dict))
    total_errors = sum(r.get('error_count', 0) for r in all_results if r.get('has_errors', False))
    
    # 生成分析报告
    analysis_report = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": "P3_optimal_template",
        "template": "Edit the following text for spelling and grammar mistakes, make minimal changes, and return only the corrected text. If the text is already correct, return it without any explanations:",
        "paper_reference": "MTSummit 2025 - arXiv:2505.06004",
        
        # 处理统计
        "processing_stats": {
            "total_reviews": total_reviews,
            "reviews_with_errors": reviews_with_errors,
            "reviews_without_errors": reviews_without_errors,
            "error_ratio": round(reviews_with_errors / total_reviews * 100, 2) if total_reviews > 0 else 0,
            "total_words": total_words,
            "total_errors_found": total_errors,
            "error_rate_per_100_words": round(total_errors / total_words * 100, 3) if total_words > 0 else 0
        },
        
        # 提取状态分布
        "extraction_status_distribution": dict(error_stats),
        
        # 详细结果
        "review_results": all_results
    }
    
    # 保存报告
    output_file = os.path.join(output_dir, f"p3_analysis_{user_id}.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Analysis saved to {output_file}")
    
    # 打印摘要
    log_with_timestamp("=" * 80)
    log_with_timestamp("P3 ANALYSIS SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Total reviews: {total_reviews}")
    log_with_timestamp(f"Reviews with errors: {reviews_with_errors} ({analysis_report['processing_stats']['error_ratio']}%)")
    log_with_timestamp(f"Reviews without errors: {reviews_without_errors}")
    log_with_timestamp(f"Total errors found: {total_errors}")
    log_with_timestamp(f"Error rate: {analysis_report['processing_stats']['error_rate_per_100_words']}/100 words")
    log_with_timestamp("=" * 80)
    
    return analysis_report


# ============================================================================
# Main函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4B: P3最优模板错误提取 (Based on MTSummit 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个用户
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID
  
  # 限制评论数量
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID \\
    --max-reviews 50
    
  # 自定义输出目录
  python 04_p3_error_extraction.py --reviews-file reviews_USER_ID.json --user-ids USER_ID \\
    --output-dir /custom/path
        """
    )
    
    parser.add_argument(
        "--reviews-file",
        required=True,
        help="Path to reviews_{USER_ID}.json file"
    )
    
    parser.add_argument(
        "--user-ids",
        required=True,
        help="User ID to process"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis",
        help="Output directory (default: %(default)s)"
    )
    
    parser.add_argument(
        "--max-reviews",
        type=int,
        help="Maximum number of reviews to analyze per user"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum concurrent workers (default: 20)"
    )
    
    args = parser.parse_args()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 4B: P3最优模板错误提取")
    log_with_timestamp("=" * 80)
    log_with_timestamp("Paper: MTSummit 2025 - arXiv:2505.06004")
    log_with_timestamp(f"Method: P3 Optimal Template")
    log_with_timestamp(f"Expected improvement over P1: +176~283% (F1 score)")
    log_with_timestamp("=" * 80)
    
    # 验证输入文件
    if not os.path.exists(args.reviews_file):
        log_with_timestamp(f"ERROR: Reviews file not found: {args.reviews_file}")
        sys.exit(1)
    
    # 加载评论数据
    log_with_timestamp(f"Loading reviews from {args.reviews_file}...")
    with open(args.reviews_file, 'r', encoding='utf-8') as f:
        reviews_data = json.load(f)
    
    # 准备用户数据
    if isinstance(reviews_data, dict) and 'reviews' in reviews_data:
        user_data = reviews_data
        user_data['user_id'] = args.user_ids
    elif isinstance(reviews_data, dict) and 'results' in reviews_data:
        user_data = {
            'user_id': args.user_ids,
            'reviews': reviews_data.get('results', [])
        }
    else:
        user_data = {
            'user_id': args.user_ids,
            'reviews': reviews_data if isinstance(reviews_data, list) else []
        }
    
    # 限制评论数量（如果指定）
    if args.max_reviews:
        reviews = user_data.get('reviews', [])
        user_data['reviews'] = reviews[:args.max_reviews]
        log_with_timestamp(f"Limited to {args.max_reviews} reviews")
    
    # 初始化LLM客户端
    llm_client = LLMClient()
    
    # 处理用户评论
    try:
        analysis_result = process_user_reviews(
            user_data,
            args.output_dir,
            llm_client,
            max_workers=args.max_workers
        )
        
        log_with_timestamp("\n✅ Processing completed successfully!")
        log_with_timestamp(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        log_with_timestamp(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
