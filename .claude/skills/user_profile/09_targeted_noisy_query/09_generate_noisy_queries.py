#!/usr/bin/env python3
"""
Stage 7 (V8): Generate Noisy Personalized Queries with ProfilingUD Style Constraints
Part of the User Profile Pipeline

This script applies user-specific writing style to personalized queries using:
1. ProfilingUD integer-count style constraints (from Stage 5)
2. Writing error analysis (from Stage 4)
3. Few-shot examples from user's actual reviews

Key Features:
- Uses integer-count constraints for linguistic features (e.g., "Nouns: 4-5")
- Generates 5 candidate queries and selects the best one
- Applies spelling/grammar patterns based on writing analysis

Input:
  - Dual queries from Stage 6 (dual_queries_*.json)
  - Writing analysis from Stage 4 (writing_analysis_*.json)
  - Layered features from Stage 5 (layered_features_*.json)

Output:
  - Noisy personalized queries with applied user style

Usage:
    python 10_generate_noisy_queries_v8.py \
        --dual-queries-dir dual_queries/ \
        --writing-analysis-dir writing_analysis/ \
        --layered-features-dir layered_features/ \
        --reviews-file all_user_reviews.json \
        --output-dir noisy_queries/
"""

PROMPT_VERSION = "v8_profilingud_integer_counts"

import json
import os
import re
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from llm_client import LLMClient

# Import spelling scorer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../08_spelling_difficulty")
try:
    spelling_scorer = __import__("08_spelling_scorer")
    SpellingDifficultyScorer = spelling_scorer.SpellingDifficultyScorer
except Exception as e:
    print(f"Loaded spelling scorer failed: {e}")
    SpellingDifficultyScorer = None


def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class ProfilingUDStyleLoader:
    """Load ProfilingUD style constraints from Stage 5 output."""

    def __init__(self, layered_features_dir: str):
        self.layered_features_dir = layered_features_dir
        self._cache = {}

    def load(self, user_id: str) -> Optional[Dict]:
        """Load layered features for a user."""
        if user_id in self._cache:
            return self._cache[user_id]

        filepath = os.path.join(self.layered_features_dir, f"layered_features_{user_id}.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[user_id] = data
            return data
        except Exception as e:
            log_with_timestamp(f"Error loading layered features for {user_id}: {e}")
            return None

    def get_style_prompt(self, user_id: str) -> str:
        """Get the pre-generated style prompt for a user."""
        data = self.load(user_id)
        if data:
            return data.get('style_prompt', '')
        return ''


class WritingAnalysisLoader:
    """Load writing analysis from Stage 4 output."""

    def __init__(self, writing_dir: str):
        self.writing_dir = writing_dir
        self._cache = {}

    def load(self, user_id: str) -> Optional[Dict]:
        """Load writing analysis for a user."""
        if user_id in self._cache:
            return self._cache[user_id]

        filepath = os.path.join(self.writing_dir, f"writing_analysis_{user_id}.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[user_id] = data
            return data
        except Exception as e:
            log_with_timestamp(f"Error loading writing analysis for {user_id}: {e}")
            return None

    def get_error_profile(self, user_id: str) -> Dict:
        """Get error profile summary for a user."""
        data = self.load(user_id)
        if not data:
            return {"total_errors": 0, "spelling": {}, "grammar": {}}

        stats = data.get('statistics', {})
        return {
            "total_errors": stats.get('total_errors', 0),
            "error_rate": stats.get('errors_per_100_words', 0),
            "spelling_total": stats.get('spelling_total', 0),
            "grammar_total": stats.get('grammar_total', 0),
            "spelling": stats.get('spelling', {}),
            "grammar": stats.get('grammar', {}),
        }


class ReviewsLoader:
    """Load user reviews for few-shot examples."""

    def __init__(self, reviews_file: str):
        self.reviews_file = reviews_file
        self._all_reviews = None
        self._cache = {}

    def _load_all(self):
        if self._all_reviews is not None:
            return

        if not os.path.exists(self.reviews_file):
            self._all_reviews = {}
            return

        try:
            with open(self.reviews_file, 'r', encoding='utf-8') as f:
                self._all_reviews = json.load(f)
        except Exception:
            self._all_reviews = {}

    def get_reviews(self, user_id: str) -> List[str]:
        """Get reviews for a user."""
        if user_id in self._cache:
            return self._cache[user_id]

        self._load_all()

        if user_id not in self._all_reviews:
            return []

        user_data = self._all_reviews[user_id]
        reviews = []
        for r in user_data.get('reviews', []):
            text = r.get('review_text') or r.get('reviewText', '')
            if text:
                reviews.append(text)

        self._cache[user_id] = reviews
        return reviews


class ProfilingUDPromptBuilder:
    """Build prompts using ProfilingUD integer-count constraints."""

    # Error type descriptions
    SPELLING_DESCRIPTIONS = {
        'Deletion': 'occasionally omits letters (e.g., "becuse" for "because")',
        'Homophone': 'sometimes confuses words that sound the same (e.g., "their/there")',
        'Substitution': 'occasionally substitutes incorrect letters',
        'Hard Word': 'sometimes struggles with difficult word spellings',
        'Suffix': 'occasionally makes mistakes with word endings (e.g., "useing")',
        'Transposition': 'sometimes swaps letters around',
        'Extra Space': 'occasionally adds unnecessary spaces (e.g., "alot")',
        'Insertion': 'sometimes inserts extra letters',
        'Capitalization': 'occasionally forgets proper capitalization',
    }

    GRAMMAR_DESCRIPTIONS = {
        'Agreement': 'sometimes has subject-verb agreement issues',
        'Hyphenation': 'occasionally makes hyphenation mistakes',
        'Pronoun': 'sometimes uses pronouns inconsistently',
        'Preposition': 'occasionally chooses wrong prepositions',
        'Homophone-Grammar': 'sometimes confuses similar-sounding grammar words',
        'Tense': 'sometimes mixes verb tenses',
        'Article': 'occasionally omits or misuses articles',
    }

    def __init__(self, style_loader: ProfilingUDStyleLoader,
                 writing_loader: WritingAnalysisLoader,
                 reviews_loader: ReviewsLoader = None,
                 scorer=None):
        self.style_loader = style_loader
        self.writing_loader = writing_loader
        self.reviews_loader = reviews_loader
        self.scorer = scorer

    def build_prompt(self, user_id: str, clean_query: str) -> str:
        """Build the style transfer prompt."""

        # Get ProfilingUD style constraints
        style_prompt = self.style_loader.get_style_prompt(user_id)

        # Get error profile
        error_profile = self.writing_loader.get_error_profile(user_id)
        error_examples = self._build_error_examples(error_profile)

        # Get few-shot examples
        few_shot_section = self._build_few_shot_section(user_id)

        vulnerable_str = ""
        if self.scorer:
            vulnerable_words = self.scorer.find_vulnerable_words(clean_query, user_profile=error_profile)
            if vulnerable_words:
                vulnerable_list = [f"'{w}' (Cognitive Difficulty: {score:.2f})" for w, score in vulnerable_words[:3]]
                vulnerable_str = f"\n## Vocabulary Vulnerability (Crucial)\nIn the original query, the following words are cognitively harder to spell:\n" + "\n".join("- " + v for v in vulnerable_list) + "\n\nIf you apply the user's spelling error patterns, you MUST prioritize these difficult words. Do NOT misspell simple, common words."

        prompt = f"""# Style Transfer Task

You are rewriting a search query to match a specific user's writing style.

## User's Linguistic Style Constraints

{style_prompt}

## Error Patterns (apply according to their relative frequencies below)

{error_examples}

{few_shot_section}
{vulnerable_str}

## Task

Generate **5 different versions** of the following search query, each applying the user's style in a different way. Then select the BEST one that most naturally matches this user's writing.

**Original Query**: {clean_query}

**Requirements for each version**:
1. Use first-person perspective ("I", "my", "me") - MANDATORY
2. Keep word count between 25-30 words
3. Preserve ALL key search terms and product attributes
4. Follow the linguistic constraints above (noun/verb/adj counts)
5. Apply error patterns proportionally to the provided Frequency % (e.g., if an error has 50% frequency, try to include it in ~half the versions)
6. IF injecting spelling errors, ONLY target the linguistically difficult words identified above (if any).

**Vary the style application**:
- Version 1: Focus on exact grammatical structure (follow the counts precisely)
- Version 2: Include the user's typical error patterns
- Version 3: Match sentence complexity and flow
- Version 4: Apply coordination style (how ideas connect)
- Version 5: Combine multiple style elements naturally

**Output format** (JSON):
{{
    "candidates": [
        {{"version": 1, "query": "...", "style_applied": "grammatical structure"}},
        {{"version": 2, "query": "...", "style_applied": "error patterns"}},
        {{"version": 3, "query": "...", "style_applied": "sentence complexity"}},
        {{"version": 4, "query": "...", "style_applied": "coordination style"}},
        {{"version": 5, "query": "...", "style_applied": "combined elements"}}
    ],
    "best_version": <1-5>,
    "reason": "Brief explanation"
}}

**Output**:"""

        return prompt

    def _build_error_examples(self, error_profile: Dict) -> str:
        """Build concrete error examples based on user's error profile, including relative frequencies."""
        spelling = error_profile.get('spelling', {})
        grammar = error_profile.get('grammar', {})
        spelling_total = error_profile.get('spelling_total', sum(spelling.values()))
        grammar_total = error_profile.get('grammar_total', sum(grammar.values()))

        if not spelling and not grammar:
            return "This user has no significant error patterns."

        examples = []

        # Top spelling errors
        if spelling and spelling_total > 0:
            examples.append("### Spelling Error Frequencies:")
            sorted_spelling = sorted(spelling.items(), key=lambda x: -x[1])[:3]
            for error_type, count in sorted_spelling:
                if error_type in self.SPELLING_DESCRIPTIONS:
                    weight = round((count / spelling_total) * 100)
                    examples.append(f"- {self.SPELLING_DESCRIPTIONS[error_type]} (Frequency: {weight}% of their spelling errors)")

        # Top grammar errors
        if grammar and grammar_total > 0:
            examples.append("\n### Grammar Error Frequencies:")
            sorted_grammar = sorted(grammar.items(), key=lambda x: -x[1])[:2]
            for error_type, count in sorted_grammar:
                if error_type in self.GRAMMAR_DESCRIPTIONS:
                    weight = round((count / grammar_total) * 100)
                    examples.append(f"- {self.GRAMMAR_DESCRIPTIONS[error_type]} (Frequency: {weight}% of their grammar errors)")

        if not examples:
            return "This user has no significant error patterns."

        return "\n".join(examples)

    def _build_few_shot_section(self, user_id: str) -> str:
        """Build few-shot examples section from user's actual reviews."""
        if not self.reviews_loader:
            return ""

        reviews = self.reviews_loader.get_reviews(user_id)
        if not reviews:
            return ""

        # Select representative reviews
        selected = self._select_few_shot_reviews(reviews, max_examples=3)

        if not selected:
            return ""

        lines = ["## Example Writing Samples (Learn from these)"]
        lines.append("")
        lines.append("Study how this user naturally writes:")
        lines.append("")

        for i, review in enumerate(selected, 1):
            # Truncate if too long
            if len(review) > 200:
                review = review[:197] + "..."
            lines.append(f"**Example {i}**: \"{review}\"")
            lines.append("")

        return "\n".join(lines)

    def _select_few_shot_reviews(self, reviews: List[str], max_examples: int = 3) -> List[str]:
        """Select representative reviews for few-shot examples."""
        # Filter and score reviews
        scored = []
        for review in reviews:
            if not review or len(review) < 30 or len(review) > 500:
                continue

            score = 0
            words = review.split()

            # Prefer moderate length
            if 50 <= len(review) <= 200:
                score += 2

            # Prefer first-person writing
            if any(w.lower() in ['i', 'my', 'me'] for w in words[:10]):
                score += 1

            # Prefer natural errors (not too perfect)
            score += min(len(re.findall(r'[,.]', review)), 3)

            scored.append((review, score))

        # Sort by score and return top
        scored.sort(key=lambda x: -x[1])
        return [r for r, s in scored[:max_examples]]


class StyleTransferEngine:
    """LLM-based style transfer engine."""

    def __init__(self, prompt_builder: ProfilingUDPromptBuilder):
        self.prompt_builder = prompt_builder
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            self.llm = LLMClient()
        return self.llm

    def transfer_style(self, user_id: str, clean_query: str, max_retries: int = 3) -> Dict:
        """Transfer style to a query using ProfilingUD constraints."""

        prompt = self.prompt_builder.build_prompt(user_id, clean_query)

        for attempt in range(max_retries + 1):
            try:
                llm = self._get_llm()
                response = llm.call(prompt, max_tokens=1024, temperature=0.7)

                parsed = self._parse_response(response)

                if parsed and 'candidates' in parsed and len(parsed['candidates']) >= 3:
                    candidates = parsed['candidates']
                    best_version = parsed.get('best_version', 1)

                    if 1 <= best_version <= len(candidates):
                        selected = candidates[best_version - 1]
                        rewritten = selected.get('query', '')
                    else:
                        rewritten = candidates[0].get('query', '')
                        best_version = 1

                    if rewritten:
                        validation = self._validate_query(rewritten)
                        return {
                            "original": clean_query,
                            "rewritten": rewritten,
                            "modified": clean_query.lower().strip() != rewritten.lower().strip(),
                            "method": f"profilingud_v8_{PROMPT_VERSION}",
                            "word_count": validation['word_count'],
                            "valid": validation['valid'],
                            "best_version": best_version,
                            "all_candidates": candidates,
                            "attempt": attempt + 1
                        }

            except Exception as e:
                log_with_timestamp(f"  Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue

        # Fallback
        return {
            "original": clean_query,
            "rewritten": clean_query,
            "modified": False,
            "method": "fallback",
            "valid": False
        }

    def _parse_response(self, response: str) -> Optional[dict]:
        """Parse JSON response."""
        try:
            if not response:
                return None

            # Extract JSON
            if "```json" in response:
                match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    response = match.group(1)
            elif "```" in response:
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    response = match.group(1)

            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None

    def _validate_query(self, query: str) -> dict:
        """Validate query meets requirements."""
        words = query.split()
        word_count = len(words)

        first_person_words = ['i', 'me', 'my', 'mine', "i'm", "i'd", "i'll", "i've"]
        has_first_person = any(w.lower() in first_person_words for w in words)

        word_count_ok = 25 <= word_count <= 30

        return {
            "word_count": word_count,
            "word_count_ok": word_count_ok,
            "has_first_person": has_first_person,
            "valid": word_count_ok and has_first_person
        }


def process_user(user_id: str,
                 dual_queries_file: str,
                 prompt_builder: ProfilingUDPromptBuilder,
                 output_dir: str):
    """Process a single user's queries."""

    log_with_timestamp(f"Processing user {user_id}...")

    # Load dual queries
    with open(dual_queries_file, 'r', encoding='utf-8') as f:
        dual_data = json.load(f)

    # Initialize engine
    engine = StyleTransferEngine(prompt_builder)

    results = []
    modified_count = 0

    if isinstance(dual_data, list):
        queries = dual_data
    else:
        queries = dual_data.get('queries', [])

    log_with_timestamp(f"  Processing {len(queries)} queries...")

    for query_data in queries:
        asin = query_data.get('asin')
        public_query = query_data.get('public_query', '')
        personalized_query = query_data.get('personalized_query', '')

        # Transfer style using ProfilingUD constraints
        public_result = engine.transfer_style(user_id, public_query)
        personalized_result = engine.transfer_style(user_id, personalized_query)

        if personalized_result['modified']:
            modified_count += 1

        results.append({
            "asin": asin,
            "public_query": {
                "original": public_query,
                "noisy": public_result['rewritten'],
                "modified": public_result['modified']
            },
            "personalized_query": {
                "original": personalized_query,
                "noisy": personalized_result['rewritten'],
                "modified": personalized_result['modified'],
                "method": personalized_result['method']
            }
        })

    # Save results
    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "total_queries": len(results),
        "modified_queries": modified_count,
        "modification_rate": round(modified_count / len(results) * 100, 1) if results else 0,
        "queries": results
    }

    output_file = os.path.join(output_dir, f"noisy_queries_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"  Modified {modified_count}/{len(results)} queries ({modified_count/len(results)*100:.1f}%)")
    log_with_timestamp(f"  Saved to {output_file}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7 (V8): Generate Noisy Queries with ProfilingUD Style Constraints"
    )
    parser.add_argument("--dual-queries-dir", required=True,
                        help="Directory with dual_queries_*.json from Stage 6")
    parser.add_argument("--writing-analysis-dir", required=True,
                        help="Directory with writing_analysis_*.json from Stage 4")
    parser.add_argument("--layered-features-dir", required=False, default=None,
                        help="Directory with layered_features_*.json from Stage 5 (optional, will skip if not provided)")
    parser.add_argument("--reviews-file", required=False,
                        help="Path to all_user_reviews.json for few-shot examples")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--user-ids", nargs="+",
                        help="Specific user IDs to process")
    parser.add_argument("--spelling-model-path", required=False, default=None,
                        help="Path to the trained spelling difficulty model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize loaders
    # If layered_features_dir is not provided, use an empty directory (style constraints will be empty)
    layered_features_dir = args.layered_features_dir if args.layered_features_dir else ""
    style_loader = ProfilingUDStyleLoader(layered_features_dir)
    writing_loader = WritingAnalysisLoader(args.writing_analysis_dir)
    reviews_loader = ReviewsLoader(args.reviews_file) if args.reviews_file else None

    scorer = None
    if SpellingDifficultyScorer:
        log_with_timestamp("Initializing target finding scorer...")
        scorer = SpellingDifficultyScorer(args.spelling_model_path)

    # Initialize prompt builder
    prompt_builder = ProfilingUDPromptBuilder(
        style_loader=style_loader,
        writing_loader=writing_loader,
        reviews_loader=reviews_loader,
        scorer=scorer
    )

    # Find users
    dual_files = {}
    for f in os.listdir(args.dual_queries_dir):
        if f.startswith("dual_queries_") and f.endswith(".json"):
            uid = f.replace("dual_queries_", "").replace(".json", "")
            dual_files[uid] = f

    if args.user_ids:
        user_ids = [u for u in args.user_ids if u in dual_files]
    else:
        user_ids = list(dual_files.keys())

    log_with_timestamp(f"Found {len(user_ids)} users to process")
    log_with_timestamp(f"Prompt version: {PROMPT_VERSION}")

    summary = []
    for user_id in user_ids:
        dual_path = os.path.join(args.dual_queries_dir, dual_files[user_id])

        result = process_user(
            user_id=user_id,
            dual_queries_file=dual_path,
            prompt_builder=prompt_builder,
            output_dir=args.output_dir
        )

        summary.append({
            "user_id": user_id,
            "total_queries": result['total_queries'],
            "modified_queries": result['modified_queries'],
            "modification_rate": result['modification_rate']
        })

    # Save summary
    summary_file = os.path.join(args.output_dir, "noisy_queries_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prompt_version": PROMPT_VERSION,
            "total_users": len(summary),
            "users": summary
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nSummary saved to {summary_file}")
    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
