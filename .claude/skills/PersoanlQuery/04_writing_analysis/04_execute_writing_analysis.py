#!/usr/bin/env python3
"""
Stage 3.5: Execute Writing Style Analysis
Part of the User Profile Pipeline

This script executes LLM-based error analysis on user reviews,
extracting spelling and grammar error patterns for each user.
The results are used to generate realistic noisy queries.

Input: Writing prompts from 08_generate_writing_prompts.py
Output: Error analysis results per user
"""

import json
import os
import sys
import re
import argparse
import concurrent.futures
from datetime import datetime
from collections import defaultdict

# Add parent directory for llm_client import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from llm_client import LLMClient

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def parse_response(response: str) -> dict:
    """Clean and parse JSON from LLM response."""
    try:
        if not response:
            return None

        json_content = response
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_content = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_content = match.group(1)
        else:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_content = match.group(0)

        return json.loads(json_content)
    except Exception as e:
        return None

def analyze_review(prompt_data: dict) -> dict:
    """Analyze a single review for errors."""
    review_idx = prompt_data.get('review_idx')
    asin = prompt_data.get('asin')
    prompt = prompt_data.get('prompt')

    result = {
        "review_idx": review_idx,
        "asin": asin,
        "spelling_errors": {},
        "grammar_errors": {},
        "status": "pending"
    }

    try:
        client = LLMClient()
        response = client.call(prompt, max_tokens=2048)
        parsed = parse_response(response)

        if parsed:
            # Filter out empty error lists
            spelling = parsed.get('spelling_errors', {})
            grammar = parsed.get('grammar_errors', {})

            result["spelling_errors"] = {k: v for k, v in spelling.items() if v}
            result["grammar_errors"] = {k: v for k, v in grammar.items() if v}
            result["status"] = "success"
        else:
            result["status"] = "parse_failed"
    except Exception as e:
        result["status"] = f"error: {str(e)}"

    return result

def compute_statistics(results: list, prompts_data: list = None) -> dict:
    """Compute error statistics from analysis results."""
    stats = {
        "spelling": defaultdict(int),
        "grammar": defaultdict(int),
        "total_reviews_analyzed": len(results),
        "spelling_total": 0,
        "grammar_total": 0,
        "total_errors": 0,
        "total_words": 0,
        "errors_per_100_words": 0.0
    }

    total_word_count = 0

    # Count words from original review texts if available
    if prompts_data:
        for prompt_item in prompts_data:
            review_text = prompt_item.get("review_text", "")
            if review_text:
                total_word_count += len(review_text.split())

    for result in results:
        # Count spelling errors by category
        for category, errors in result.get("spelling_errors", {}).items():
            stats["spelling"][category] += len(errors)
            stats["spelling_total"] += len(errors)

        # Count grammar errors by category
        for category, errors in result.get("grammar_errors", {}).items():
            stats["grammar"][category] += len(errors)
            stats["grammar_total"] += len(errors)

    stats["total_errors"] = stats["spelling_total"] + stats["grammar_total"]
    stats["total_words"] = total_word_count

    # Calculate errors per 100 words
    if total_word_count > 0:
        stats["errors_per_100_words"] = round(stats["total_errors"] / total_word_count * 100, 2)

    # Convert defaultdict to regular dict
    stats["spelling"] = dict(stats["spelling"])
    stats["grammar"] = dict(stats["grammar"])

    return stats

def main():
    parser = argparse.ArgumentParser(description="Stage 3.5: Execute Writing Style Analysis")
    parser.add_argument("--prompts-dir", required=True, help="Directory containing writing_prompts_*.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis results")
    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent LLM calls")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all prompt files
    prompt_files = [f for f in os.listdir(args.prompts_dir)
                    if f.startswith("writing_prompts_") and f.endswith(".json")]

    log_with_timestamp(f"Found {len(prompt_files)} prompt files to process")

    for prompt_file in prompt_files:
        user_id = prompt_file.replace("writing_prompts_", "").replace(".json", "")
        prompt_path = os.path.join(args.prompts_dir, prompt_file)

        log_with_timestamp(f"Processing user {user_id}...")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = data.get("prompts", [])
        results = []

        log_with_timestamp(f"  Analyzing {len(prompts)} reviews with {args.max_workers} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_prompt = {executor.submit(analyze_review, p): p for p in prompts}

            completed = 0
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        log_with_timestamp(f"  Progress: {completed}/{len(prompts)} reviews analyzed")
                except Exception as e:
                    log_with_timestamp(f"  Error analyzing review: {e}")

        # Compute statistics (pass prompts data for word counting)
        stats = compute_statistics(results, prompts)

        # Save analysis results
        output_file = os.path.join(args.output_dir, f"writing_analysis_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "total_reviews": len(results),
                "statistics": stats,
                "results": results
            }, f, indent=2, ensure_ascii=False)

        log_with_timestamp(f"  Completed: {stats['spelling_total']} spelling errors, {stats['grammar_total']} grammar errors")
        log_with_timestamp(f"  Saved to {output_file}")

    log_with_timestamp("All users processed!")

if __name__ == "__main__":
    main()
