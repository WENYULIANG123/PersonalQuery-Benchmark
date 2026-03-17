#!/usr/bin/env python3
"""
Stage 4B: Grammar Error Detection using BERT-GED Model

使用预训练的 BERT-GED 模型检测用户评论中的语法错误。
基于 HuggingFace 的 sahilnishad/BERT-GED-FCE-FT 模型。

Input: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json

Output: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/grammar_analysis_{user_id}.json
  - Summary statistics per user

Usage:
  # Process specific user
  python 04_grammar_error_detection.py --reviews-file reviews_USER_ID.json --user-ids USER_ID
  
  # Process with custom output directory
  python 04_grammar_error_detection.py --reviews-file reviews_USER_ID.json --output-dir /path/to/output
"""

import json
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

def log_with_timestamp(message: str):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class BERTGrammarDetector:
    """使用BERT-GED模型检测语法错误"""
    
    def __init__(self, model_id: str = "sahilnishad/BERT-GED-FCE-FT", min_error_probability: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_error_probability = min_error_probability
        self.model_id = model_id
        
        log_with_timestamp(f"Initializing BERT-GED model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        log_with_timestamp(f"Model loaded successfully")
    
    def detect_errors(self, text: str) -> List[Dict]:
        """
        检测单条文本中的语法错误
        
        Args:
            text: 输入文本
            
        Returns:
            错误列表，每个错误包含位置、标记、概率等信息
        """
        tokens = text.split()
        
        if not tokens:
            return []
        
        encodings = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        word_ids = encodings.word_ids()
        
        errors = []
        
        for word_idx, token in enumerate(tokens):
            token_mask = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if token_mask:
                token_logits = logits[0, token_mask[0], :]
                token_probs = F.softmax(token_logits, dim=-1)
                
                incorrect_prob = token_probs[1].item()
                correct_prob = token_probs[0].item()
                
                if incorrect_prob >= self.min_error_probability:
                    errors.append({
                        "token_idx": word_idx,
                        "token": token,
                        "prediction": "INCORRECT",
                        "incorrect_probability": float(incorrect_prob),
                        "correct_probability": float(correct_prob),
                        "error_type": "GRAMMAR"
                    })
        
        return errors


def load_reviews(reviews_file: str) -> List[str]:
    """
    加载用户的评论数据
    
    Args:
        reviews_file: reviews_{USER_ID}.json 文件路径
        
    Returns:
        评论文本列表
    """
    with open(reviews_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    comments = []
    results = data.get('results', [])
    
    for item in results:
        if isinstance(item, dict):
            target_review = item.get('target_review', '')
            other_reviews = item.get('other_reviews', [])
            
            if target_review and target_review.strip():
                comments.append(target_review.strip())
            
            for review in other_reviews:
                if isinstance(review, str) and review.strip():
                    comments.append(review.strip())
        elif isinstance(item, str) and item.strip():
            comments.append(item.strip())
    
    return comments


def analyze_user_reviews(
    reviews_file: str,
    user_id: str,
    output_dir: str,
    max_reviews: Optional[int] = None
) -> Dict:
    """
    分析单个用户的所有评论
    
    Args:
        reviews_file: 评论文件路径
        user_id: 用户ID
        output_dir: 输出目录
        max_reviews: 最多处理的评论数
        
    Returns:
        分析结果统计字典
    """
    log_with_timestamp(f"[{user_id}] Loading reviews from {reviews_file}...")
    
    comments = load_reviews(reviews_file)
    
    if max_reviews:
        comments = comments[:max_reviews]
    
    log_with_timestamp(f"[{user_id}] Loaded {len(comments)} reviews")
    
    detector = BERTGrammarDetector()
    
    analysis_results = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "reviews_analyzed": len(comments),
        "total_words": 0,
        "total_grammar_errors": 0,
        "grammar_error_rate": 0.0,
        "reviews_with_errors": 0,
        "error_distribution": defaultdict(int),
        "review_details": []
    }
    
    log_with_timestamp(f"[{user_id}] Analyzing reviews for grammar errors...")
    
    for idx, comment in enumerate(comments):
        if (idx + 1) % max(1, len(comments) // 10) == 0:
            log_with_timestamp(f"[{user_id}] Progress: {idx + 1}/{len(comments)}")
        
        words = comment.split()
        word_count = len(words)
        analysis_results["total_words"] += word_count
        
        errors = detector.detect_errors(comment)
        
        if errors:
            analysis_results["reviews_with_errors"] += 1
            analysis_results["total_grammar_errors"] += len(errors)
            
            review_detail = {
                "review_idx": idx,
                "text": comment[:200],
                "word_count": word_count,
                "error_count": len(errors),
                "errors": errors[:5]
            }
            analysis_results["review_details"].append(review_detail)
            
            for error in errors:
                analysis_results["error_distribution"][error["error_type"]] += 1
    
    if analysis_results["total_words"] > 0:
        analysis_results["grammar_error_rate"] = round(
            analysis_results["total_grammar_errors"] / analysis_results["total_words"] * 100,
            2
        )
    
    analysis_results["error_distribution"] = dict(analysis_results["error_distribution"])
    analysis_results["review_details"] = analysis_results["review_details"][:20]
    
    log_with_timestamp(f"[{user_id}] Detected {analysis_results['total_grammar_errors']} grammar errors")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"grammar_analysis_{user_id}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"[{user_id}] Results saved to {output_file}")
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Detect grammar errors in user reviews using BERT-GED model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--reviews-file",
        required=True,
        help="Path to reviews_{USER_ID}.json file"
    )
    
    parser.add_argument(
        "--user-ids",
        required=True,
        help="User ID for processing"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis",
        help="Output directory (default: %(default)s)"
    )
    
    parser.add_argument(
        "--max-reviews",
        type=int,
        help="Maximum number of reviews to analyze"
    )
    
    parser.add_argument(
        "--min-error-probability",
        type=float,
        default=0.5,
        help="Minimum probability threshold for error detection (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    log_with_timestamp("="*80)
    log_with_timestamp("Stage 4B: Grammar Error Detection using BERT-GED")
    log_with_timestamp("="*80)
    
    if not os.path.exists(args.reviews_file):
        log_with_timestamp(f"ERROR: Reviews file not found: {args.reviews_file}")
        sys.exit(1)
    
    try:
        result = analyze_user_reviews(
            reviews_file=args.reviews_file,
            user_id=args.user_ids,
            output_dir=args.output_dir,
            max_reviews=args.max_reviews
        )
        
        log_with_timestamp("="*80)
        log_with_timestamp("ANALYSIS SUMMARY")
        log_with_timestamp("="*80)
        log_with_timestamp(f"User ID: {result['user_id']}")
        log_with_timestamp(f"Reviews analyzed: {result['reviews_analyzed']}")
        log_with_timestamp(f"Total words: {result['total_words']}")
        log_with_timestamp(f"Total grammar errors: {result['total_grammar_errors']}")
        log_with_timestamp(f"Grammar error rate: {result['grammar_error_rate']}/100 words")
        log_with_timestamp(f"Reviews with errors: {result['reviews_with_errors']}/{result['reviews_analyzed']}")
        
        if result['error_distribution']:
            log_with_timestamp(f"Error types: {result['error_distribution']}")
        
        log_with_timestamp("="*80)
        log_with_timestamp("PROCESSING COMPLETE!")
        log_with_timestamp("="*80)
        
    except Exception as e:
        log_with_timestamp(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
