#!/usr/bin/env python3
"""
Stage 4: Generate Writing Style Analysis Prompts
Part of the User Profile Pipeline

This script generates prompts for analyzing user writing patterns,
including spelling errors (9 types, aligned with Stage 8 model) and grammar errors (7 types).
The output is used to create realistic noisy queries.

Input: User reviews from user_product_reviews.json
Output: Writing analysis prompts for each user
"""

import json
import os
import argparse
from datetime import datetime

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def create_error_analysis_prompt(review_text: str) -> str:
    """Create a prompt for error analysis."""
    return f"""<s> [INST] ## Task: Error Classification - Spelling & Grammar (Arts & Crafts Domain)

**Input:** "{review_text}"

**Goal:** Identify GENUINE spelling errors AND grammar errors in this ENTIRE review text. Return JSON only.

### IGNORE (DO NOT REPORT AS ERRORS):
1. **Punctuation Only:** Missing commas, periods (unless they create grammar errors).
2. **Capitalization Only:** Case differences only.
3. **Brand Names:** If you think this might be a brand name (e.g., Sizzix, Cuttlebug), do not report it.
4. **Stylistic Choices:** Informal but acceptable expressions (e.g., "pretty good" vs "very good").

---

### SPELLING ERROR CATEGORIES (9 Types)
1. **Deletion** (Missing letters): `colr` -> `color`
2. **Insertion** (Extra letters): `accross` -> `across`
3. **Transposition** (Swapped letters): `teh` -> `the`
4. **Scramble** (Complex rearrangement): `definitly` -> `definitely`
5. **Substitution** (Wrong letter): `wprk` -> `work`
6. **Homophone** (Sound-alike): `there` -> `their`
7. **Suffix** (Ending wrong): `runing` -> `running`
8. **Hard Word** (Difficult spelling): `fuchsia` -> `fushia`
9. **Extra Space** (Unnecessary space/hyphen): `note book` -> `notebook`, `note-book` -> `notebook`

### GRAMMAR ERROR CATEGORIES (7 Types)
1. **Agreement** (Subject-verb, number consistency): `it is` -> `they are` (for plural subject)
2. **Collocation** (Unnatural word pairing): `between 4 or 5` -> `between 4 and 5`
3. **Preposition** (Missing/wrong preposition): `excel that` -> `excel at that`
4. **Pronoun** (Wrong relative pronoun): `what I consider` -> `which I consider`
5. **Suffix** (Wrong word form): `more fine` -> `finer`
6. **Homophone-Grammar** (Verb tense/form): `lay down` -> `lie down`
7. **Hyphenation** (Missing hyphen): `good size` -> `good-sized`

---

### OUTPUT JSON structure:
```json
{{
  "spelling_errors": {{
    "Deletion": [ {{ "original": "...", "corrected": "...", "fragment": "...", "reason": "..." }} ],
    "Insertion": [], "Transposition": [], "Scramble": [], "Substitution": [],
    "Homophone": [], "Suffix": [], "Hard Word": [], "Extra Space": []
  }},
  "grammar_errors": {{
    "Agreement": [ {{ "original": "...", "corrected": "...", "fragment": "...", "reason": "..." }} ],
    "Collocation": [], "Preposition": [], "Pronoun": [],
    "Suffix": [], "Homophone": [], "Hyphenation": []
  }}
}}
```
[/INST]"""

def main():
    parser = argparse.ArgumentParser(description="Stage 3.5: Generate Writing Style Analysis Prompts")
    parser.add_argument("--reviews-file",
                        default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json",
                        help="Path to user reviews JSON file")
    parser.add_argument("--user-ids", nargs="+", required=True, help="User IDs to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for prompts")
    parser.add_argument("--num-reviews", type=int, default=None, help="Number of reviews to analyze per user (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load reviews
    log_with_timestamp(f"Loading reviews from {args.reviews_file}...")
    with open(args.reviews_file, 'r', encoding='utf-8') as f:
        all_reviews = json.load(f)

    for user_id in args.user_ids:
        # Handle two formats:
        # 1. User-keyed dict: {"A1BBCMQSEJN0PP": {"reviews": [...]}}
        # 2. Single user format: {"user_id": "A1BBCMQSEJN0PP", "reviews": [...]}
        if user_id in all_reviews:
            user_data = all_reviews[user_id].get('reviews', [])
        elif all_reviews.get('user_id') == user_id:
            user_data = all_reviews.get('reviews', [])
        else:
            log_with_timestamp(f"Warning: User {user_id} not found in reviews file, skipping...")
            continue

        reviews_to_process = user_data[:args.num_reviews] if args.num_reviews else user_data

        log_with_timestamp(f"Processing {len(reviews_to_process)} reviews for user {user_id}...")

        prompts = []
        for idx, review in enumerate(reviews_to_process):
            # Try both field names: reviewText (camelCase) and review_text (snake_case)
            text = review.get('reviewText', '') or review.get('review_text', '')
            text = text.strip()
            if not text:
                continue

            prompts.append({
                "review_idx": idx,
                "asin": review.get('asin', ''),
                "prompt": create_error_analysis_prompt(text)
            })

        # Save prompts
        output_file = os.path.join(args.output_dir, f"writing_prompts_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "total_prompts": len(prompts),
                "prompts": prompts
            }, f, indent=2, ensure_ascii=False)

        log_with_timestamp(f"Generated {len(prompts)} prompts for user {user_id} -> {output_file}")

    log_with_timestamp("Done!")

if __name__ == "__main__":
    main()
