#!/usr/bin/env python3
"""
Generate Error Analysis Prompts for Manual User Error Pattern Extraction.
Identifies both spelling errors (10 types) and grammar errors (7 types).
"""

import json
import os
import argparse

def create_prompt(review_text):
    return f"""<s> [INST] ## Task: Error Classification - Spelling & Grammar (Arts & Crafts Domain)

**Input:** "{review_text}"

**Goal:** Identify GENUINE spelling errors AND grammar errors in this ENTIRE review text. Return JSON only.

### 🚫 IGNORE (DO NOT REPORT AS ERRORS):
1. **Punctuation Only:** Missing commas, periods (unless they create grammar errors).
2. **Capitalization Only:** Case differences only.
3. **Brand Names:** If you think this might be a brand name (e.g., Sizzix, Cuttlebug), do not report it.
4. **Stylistic Choices:** Informal but acceptable expressions (e.g., "pretty good" vs "very good").

---

### 📂 SPELLING ERROR CATEGORIES (10 Types)
1. **Deletion** (Missing letters): `colr` -> `color`
2. **Insertion** (Extra letters): `accross` -> `across`
3. **Transposition** (Swapped letters): `teh` -> `the`
4. **Scramble** (Complex rearrangement): `definitly` -> `definitely`
5. **Substitution** (Wrong letter): `wprk` -> `work`
6. **Homophone** (Sound-alike): `there` -> `their`
7. **Suffix** (Ending wrong): `runing` -> `running`
8. **Hard Word** (Difficult spelling): `fuchsia` -> `fushia`
9. **Extra Space** (Compound words): `note book` -> `notebook`
10. **Extra Hyphen** (Unnecessary hyphen): `note-book` -> `notebook`

### 📂 GRAMMAR ERROR CATEGORIES (7 Types)
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
    "Homophone": [], "Suffix": [], "Hard Word": [], "Extra Space": [], "Extra Hyphen": []
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--num_reviews", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    with open(args.input, 'r') as f:
        data = json.load(f)

    if args.user_id not in data:
        print(f"❌ User ID {args.user_id} not found.")
        return

    reviews = data[args.user_id].get('reviews', [])
    reviews_to_process = reviews[:args.num_reviews] if args.num_reviews else reviews

    output_data = []
    for idx, review in enumerate(reviews_to_process):
        text = review.get('review_text', '').strip()
        if not text: continue
        output_data.append({"review_idx": idx, "prompt": create_prompt(text)})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(output_data)} error analysis prompts for user {args.user_id} to {args.output}")

if __name__ == "__main__":
    main()
