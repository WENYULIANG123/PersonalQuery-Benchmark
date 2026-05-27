#!/usr/bin/env python3
"""
Stage 4: P3 Optimal Template Comprehensive Error Analysis

Complete P3 error analysis pipeline combining:
1. Error extraction using P3 optimal template (MTSummit 2025, arXiv:2505.06004)
2. Detailed error recognition with position, type, and classification

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/writing_analysis_{user_id}.json

Usage:
  python 04_p3_comprehensive_analysis.py --reviews-file reviews_USER_ID.json --user-ids USER_ID
  python 04_p3_comprehensive_analysis.py --all-users
"""

import json
import os
import sys
import argparse
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


# ============================================================================
# P3 Error Extraction
# ============================================================================

class P3ErrorExtractor:
    """Extract errors using P3 optimal template"""
    
    P3_TEMPLATE = """Edit the following text for spelling and grammar mistakes, make minimal changes, and return only the corrected text. If the text is already correct, return it without any explanations:."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def create_p3_prompt(self, review_text: str) -> str:
        return f"""<s>[INST] {self.P3_TEMPLATE}

"{review_text}"

Please return ONLY the corrected text. If no corrections are needed, return the original text exactly as it is.
[/INST]"""
    
    def extract_errors(self, original_text: str, max_retries: int = 5) -> Dict:
        prompt = self.create_p3_prompt(original_text)
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.call(
                    prompt=prompt,
                    max_tokens=256
                )
                
                corrected_text = response.strip()
                
                return {
                    "status": "success",
                    "original": original_text,
                    "corrected": corrected_text,
                    "has_errors": original_text != corrected_text
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        "status": "error",
                        "original": original_text,
                        "corrected": original_text,
                        "error": str(e),
                        "has_errors": False
                    }


# ============================================================================
# Error Type Classification
# ============================================================================

class ErrorTypeClassifier:
    """Automatic error type classification"""
    
    PUNCTUATION = set('.,;:!?\'"()-[]{}""''…«»')
    
    @staticmethod
    def classify(original: str, corrected: str) -> dict:
        if original.lower() == corrected.lower() and original != corrected:
            return {
                "type": "capitalization",
                "description": f"Capitalization: '{original}' → '{corrected}'",
                "confidence": 0.95
            }
        
        if ErrorTypeClassifier._is_punctuation_change(original, corrected):
            return {
                "type": "punctuation",
                "description": ErrorTypeClassifier._describe_punctuation_change(original, corrected),
                "confidence": 0.95
            }
        
        if original.replace(" ", "") == corrected.replace(" ", ""):
            return {
                "type": "whitespace",
                "description": f"Whitespace adjustment: '{original}' → '{corrected}'",
                "confidence": 0.9
            }
        
        if original.replace("-", "") == corrected.replace("-", ""):
            return {
                "type": "formatting",
                "description": f"Hyphenation: '{original}' → '{corrected}'",
                "confidence": 0.9
            }
        
        if ErrorTypeClassifier._is_quote_change(original, corrected):
            return {
                "type": "formatting",
                "description": f"Quote style: '{original}' → '{corrected}'",
                "confidence": 0.85
            }
        
        if len(original) > 1 and len(corrected) > 1:
            edit_dist = ErrorTypeClassifier._edit_distance(original.lower(), corrected.lower())
            if edit_dist <= 2:
                return {
                    "type": "spelling",
                    "description": f"Spelling: '{original}' → '{corrected}'",
                    "confidence": 0.8
                }
            elif ErrorTypeClassifier._is_morphological_change(original, corrected):
                return {
                    "type": "grammar",
                    "description": f"Grammar: '{original}' → '{corrected}'",
                    "confidence": 0.8
                }
        
        return {
            "type": "grammar",
            "description": f"Correction: '{original}' → '{corrected}'",
            "confidence": 0.7
        }
    
    @staticmethod
    def _is_punctuation_change(original: str, corrected: str) -> bool:
        orig_punct = "".join(c for c in original if c in ErrorTypeClassifier.PUNCTUATION)
        corr_punct = "".join(c for c in corrected if c in ErrorTypeClassifier.PUNCTUATION)
        orig_alpha = "".join(c for c in original if c.isalnum() or c.isspace())
        corr_alpha = "".join(c for c in corrected if c.isalnum() or c.isspace())
        
        return (orig_alpha == corr_alpha and orig_punct != corr_punct)
    
    @staticmethod
    def _describe_punctuation_change(original: str, corrected: str) -> str:
        orig_punct = set(c for c in original if c in ErrorTypeClassifier.PUNCTUATION)
        corr_punct = set(c for c in corrected if c in ErrorTypeClassifier.PUNCTUATION)
        
        added = corr_punct - orig_punct
        removed = orig_punct - corr_punct
        
        desc = "Punctuation"
        if added:
            desc += f", added: {repr(list(added)[0])}" if len(added) == 1 else f", added: {repr(list(added))}"
        if removed:
            desc += f", removed: {repr(list(removed)[0])}" if len(removed) == 1 else f", removed: {repr(list(removed))}"
        
        return desc
    
    @staticmethod
    def _is_quote_change(original: str, corrected: str) -> bool:
        quote_chars = {'"', "'", """, """, "'", "'", "«", "»"}
        orig_has_quote = any(c in original for c in quote_chars)
        corr_has_quote = any(c in corrected for c in quote_chars)
        return orig_has_quote and corr_has_quote
    
    @staticmethod
    def _is_morphological_change(original: str, corrected: str) -> bool:
        suffixes = ['ed', 'ing', 'ly', 's', 'es', 'er', 'est', 'tion', 'ment', 'able', 'ible']
        
        orig_lower = original.lower()
        corr_lower = corrected.lower()
        
        for suffix in suffixes:
            if orig_lower.endswith(suffix) != corr_lower.endswith(suffix):
                return True
        
        return False
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return ErrorTypeClassifier._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# ============================================================================
# Detailed Error Analysis
# ============================================================================

class DetailedErrorExtractor:
    """Extract detailed error information"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
    
    def extract_errors(self, original: str, corrected: str) -> list:
        errors = []
        error_id = 1
        
        matcher = difflib.SequenceMatcher(None, original, corrected)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            orig_text = original[i1:i2]
            corr_text = corrected[j1:j2]
            
            context_start = max(0, i1 - self.window_size)
            context_end = min(len(original), i2 + self.window_size)
            orig_context = original[context_start:context_end]
            
            context_start_corr = max(0, j1 - self.window_size)
            context_end_corr = min(len(corrected), j2 + self.window_size)
            corr_context = corrected[context_start_corr:context_end_corr]
            
            error_type_info = ErrorTypeClassifier.classify(orig_text, corr_text)
            
            error = {
                "error_id": error_id,
                "type": error_type_info["type"],
                "description": error_type_info["description"],
                "confidence": error_type_info["confidence"],
                "location": {
                    "original_position": i1,
                    "corrected_position": j1,
                    "original_snippet": self._get_snippet(original, i1, i2),
                    "corrected_snippet": self._get_snippet(corrected, j1, j2),
                },
                "details": {
                    "original": orig_text,
                    "corrected": corr_text,
                    "change_type": tag,
                    "original_length": len(orig_text),
                    "corrected_length": len(corr_text),
                },
                "context": {
                    "original": orig_context,
                    "corrected": corr_context,
                }
            }
            
            errors.append(error)
            error_id += 1
        
        return errors
    
    @staticmethod
    def _get_snippet(text: str, start: int, end: int, padding: int = 30) -> dict:
        snippet_start = max(0, start - padding)
        snippet_end = min(len(text), end + padding)
        
        snippet = text[snippet_start:snippet_end]
        
        rel_start = start - snippet_start
        rel_end = rel_start + (end - start)
        
        return {
            "text": snippet,
            "modification_start": rel_start,
            "modification_end": rel_end,
        }


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

class P3ComprehensiveAnalyzer:
    """Complete P3 analysis pipeline"""
    
    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path("/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis")
        self.reviews_dir = reviews_dir or Path("/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation")
        self.llm_client = LLMClient()
        self.p3_extractor = P3ErrorExtractor(self.llm_client)
        self.error_extractor = DetailedErrorExtractor()
    
    def process_user(self, user_id: str, reviews_file: Optional[str] = None, max_reviews: Optional[int] = None) -> dict:
        """Process single user: extract P3 errors and perform detailed analysis"""
        
        if not reviews_file:
            reviews_file = str(self.reviews_dir / f"reviews_{user_id}.json")
        
        if not os.path.exists(reviews_file):
            logger.error(f"Reviews file not found: {reviews_file}")
            return {"user_id": user_id, "status": "failed", "reason": "reviews_file_not_found"}
        
        try:
            with open(reviews_file, 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
            
            # 获取产品列表，并从target_reviews中扁平化为单个review文本列表
            products = reviews_data.get('results', reviews_data.get('reviews', []))
            
            # 构建扁平化的reviews列表：(review_text, asin)元组
            # 仅提取target_reviews（用户自己写的评论），不包含other_reviews（其他用户的评论）
            flattened_reviews = []
            for product in products:
                asin = product.get('asin', '')
                
                # 获取该产品的所有target_reviews（该用户写的评论）
                target_reviews = product.get('target_reviews', [])
                for review_text in target_reviews:
                    if isinstance(review_text, str):
                        flattened_reviews.append((review_text, asin))
            
            # 应用max_reviews限制
            if max_reviews:
                flattened_reviews = flattened_reviews[:max_reviews]
            
            detailed_results = []
            total_errors = 0
            error_type_counts = defaultdict(int)
            
            # 遍历扁平化后的reviews
            for review_idx, (review_text, asin) in enumerate(flattened_reviews):
                original = review_text  # 直接使用字符串，无需.get()
                
                p3_result = self.p3_extractor.extract_errors(original)
                
                if p3_result["status"] != "success":
                    logger.warning(f"[{user_id}] Review {review_idx}: {p3_result.get('error', 'Unknown error')}")
                    continue
                
                corrected = p3_result["corrected"]
                
                errors = self.error_extractor.extract_errors(original, corrected)
                
                for error in errors:
                    error_type_counts[error["type"]] += 1
                
                detailed_results.append({
                    "review_idx": review_idx,
                    "asin": asin,
                    "has_errors": len(errors) > 0,
                    "total_errors": len(errors),
                    "original_length": len(original),
                    "corrected_length": len(corrected),
                    "errors": errors
                })
                
                total_errors += len(errors)
                
                if (review_idx + 1) % 10 == 0:
                    logger.info(f"[{user_id}] Progress: {review_idx + 1}/{len(flattened_reviews)} reviews, {total_errors} errors found so far")
            
            error_type_percentages = {}
            for etype, count in error_type_counts.items():
                error_type_percentages[etype] = round(100 * count / total_errors, 1) if total_errors > 0 else 0
            
            output_data = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "p3_optimal_template_comprehensive",
                "paper_reference": "MTSummit 2025 - arXiv:2505.06004",
                "total_reviews": len(flattened_reviews),
                "reviews_with_errors": sum(1 for r in detailed_results if r["has_errors"]),
                "reviews_without_errors": sum(1 for r in detailed_results if not r["has_errors"]),
                "total_errors": total_errors,
                "error_type_distribution": dict(error_type_counts),
                "error_type_percentages": error_type_percentages,
                "detailed_errors": detailed_results
            }
            
            output_file = self.analysis_dir / f"writing_analysis_{user_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ [{user_id}] P3 comprehensive analysis completed: {len(flattened_reviews)} reviews, {total_errors} errors")
            
            return {
                "user_id": user_id,
                "status": "success",
                "reviews_processed": len(flattened_reviews),
                "total_errors": total_errors,
                "error_types": dict(error_type_counts),
                "output_file": str(output_file)
            }
        
        except Exception as e:
            logger.error(f"❌ [{user_id}] Processing failed: {str(e)}")
            return {"user_id": user_id, "status": "failed", "reason": str(e)}


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="P3 Optimal Template Comprehensive Error Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_p3_comprehensive_analysis.py --reviews-file reviews_USER_ID.json --user-ids USER_ID
  python 04_p3_comprehensive_analysis.py --all-users
  python 04_p3_comprehensive_analysis.py --user-ids A2GJX2KCUSR0EI A1GYEGLX3P2Y7P
        """
    )
    
    parser.add_argument("--analysis-dir", type=Path, default=Path("/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis"))
    parser.add_argument("--reviews-dir", type=Path, default=Path("/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation"))
    parser.add_argument("--reviews-file", type=str, help="Reviews file path")
    parser.add_argument("--user-ids", type=str, nargs='+', help="User IDs to process")
    parser.add_argument("--all-users", action="store_true", help="Process all users from reviews directory")
    parser.add_argument("--max-reviews", type=int, help="Max reviews per user")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent workers")
    
    args = parser.parse_args()
    
    user_ids = []
    
    if args.user_ids:
        user_ids = args.user_ids
    elif args.all_users:
        analysis_files = list(args.reviews_dir.glob("reviews_*.json"))
        user_ids = [f.stem.replace("reviews_", "") for f in analysis_files]
        user_ids = sorted(set(user_ids))
    else:
        parser.print_help()
        return
    
    logger.info("=" * 80)
    logger.info("Stage 4: P3 Optimal Template Comprehensive Error Analysis")
    logger.info("=" * 80)
    logger.info(f"Processing {len(user_ids)} users (sequential)")
    logger.info(f"Concurrent workers per user: {args.max_workers}")
    logger.info("=" * 80)
    
    analyzer = P3ComprehensiveAnalyzer(args.analysis_dir, args.reviews_dir)
    
    results = []
    for user_idx, user_id in enumerate(user_ids, 1):
        logger.info(f"[{user_idx}/{len(user_ids)}] Processing user: {user_id}")
        result = analyzer.process_user(user_id, args.reviews_file, args.max_reviews)
        results.append(result)
        
        if result["status"] == "success":
            logger.info(f"  ✓ {result['user_id']}: {result['reviews_processed']} reviews, {result['total_errors']} errors")
        else:
            logger.warning(f"  ✗ {result['user_id']}: {result.get('reason', 'Unknown error')}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    logger.info("=" * 80)
    logger.info(f"Completed: {len(successful)} success, {len(failed)} failed")
    logger.info("=" * 80)
    
    if successful:
        total_reviews = sum(r["reviews_processed"] for r in successful)
        total_errors = sum(r["total_errors"] for r in successful)
        logger.info(f"Total: {total_reviews} reviews, {total_errors} errors")
        
        all_error_types = defaultdict(int)
        for r in successful:
            for etype, count in r["error_types"].items():
                all_error_types[etype] += count
        
        logger.info("Error type distribution:")
        for etype, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
