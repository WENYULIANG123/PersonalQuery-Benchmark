#!/usr/bin/env python3
"""
Stage 4.5: Generate Noisy Personalized Queries
Part of the User Profile Pipeline

This script applies user-specific writing style errors (spelling & grammar)
to personalized queries, creating realistic noisy versions.

Input:
  - Dual queries from Stage 4 (dual_queries_*.json)
  - Writing analysis from Stage 3.5 (writing_analysis_*.json)

Output:
  - Noisy personalized queries with applied errors
"""

import json
import os
import re
import argparse
from datetime import datetime
from collections import defaultdict

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# Try to import spacy for grammar analysis
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    log_with_timestamp("Warning: spaCy not available, grammar analysis will be limited")


class NoisyQueryGenerator:
    """
    Generate noisy queries based on user writing style analysis.

    Key principles:
    1. Single error per query (to maintain naturalness)
    2. Error weights based on user's actual error patterns
    3. Dynamic threshold based on error rate
    """

    def __init__(self, writing_analysis_file: str):
        """Initialize with writing analysis results."""
        self.lexical_map = {}
        self.grammar_weights = {
            "agreement": 0.30,
            "preposition": 0.30,
            "suffix": 0.30,
            "hyphenation": 0.30,
            "collocation": 0.30,
        }
        self.error_rate = 0.0
        self.threshold = 0.08

        # Default error patterns
        self.default_patterns = {
            # Spelling errors (10 types)
            "deletion": [
                ("color", "colr"), ("beautiful", "beatiful"), ("every", "evry"),
                ("about", "abot"), ("different", "diferent")
            ],
            "insertion": [
                ("across", "accross"), ("really", "realley"), ("beautiful", "beautifull"),
                ("necessary", "neccesary"), ("until", "untill")
            ],
            "transposition": [
                ("the", "teh"), ("their", "thier"), ("from", "form"),
                ("with", "wiht"), ("that", "taht")
            ],
            "scramble": [
                ("definitely", "definitly"), ("separate", "seperate"),
                ("beginning", "begining"), ("receive", "recieve")
            ],
            "substitution": [
                ("work", "wprk"), ("these", "thsee"), ("good", "godo")
            ],
            "homophone": [
                ("their", "there"), ("your", "youre"), ("palette", "pallet"),
                ("intact", "in tact")
            ],
            "suffix": [
                ("running", "runing"), ("boxes", "boxs"), ("easily", "easyly")
            ],
            "hard_word": [
                ("fuchsia", "fuschia"), ("necessary", "neccesary"),
                ("accommodate", "accomodate")
            ],
            "extra_space": [
                ("notebook", "note book"), ("intact", "in tact"),
                ("background", "back ground")
            ],
            "extra_hyphen": [
                ("notebook", "note-book"), ("today", "to-day")
            ],
            # Grammar errors (7 types)
            "agreement": [
                ("are", "is"), ("do", "does"), ("have", "has"),
                ("were", "was"), ("these are", "this is")
            ],
            "collocation": [
                ("between 4 or 5", "between 4 and 5"),
                ("fit in", "fit into"), ("based off", "based on")
            ],
            "preposition": [
                ("excel at", "excel"), ("slots for", "slots"),
                ("range from", "range of")
            ],
            "pronoun": [
                ("which I", "what I"), ("that is", "which is")
            ],
            "grammar_suffix": [
                ("more fine", "finer"), ("to using", "to use"),
                ("most big", "biggest")
            ],
            "homophone_grammar": [
                ("lie down", "lay down")
            ],
            "hyphenation": [
                ("high-quality", "high quality"), ("off-white", "off white"),
                ("good-sized", "good size")
            ]
        }

        if writing_analysis_file:
            self._load_writing_analysis(writing_analysis_file)

    def _load_writing_analysis(self, filepath: str):
        """Load writing analysis and compute error weights."""
        log_with_timestamp(f"Loading writing analysis from {filepath}...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stats = data.get('statistics', {})
            spelling_stats = stats.get('spelling', {})
            grammar_stats = stats.get('grammar', {})
            spelling_total = stats.get('spelling_total', 1)
            grammar_total = stats.get('grammar_total', 1)
            total_reviews = stats.get('total_reviews_analyzed', 1)

            log_with_timestamp(f"  Spelling errors: {spelling_total}, Grammar errors: {grammar_total}")

            # Compute weights for each error type
            # Formula: weight = 0.3 + (count / total) * 0.6, max = 0.95

            # Process spelling errors
            for err_type, count in spelling_stats.items():
                weight = min(0.95, 0.3 + (count / max(spelling_total, 1)) * 0.6)
                self._map_spelling_error(err_type, weight)

            # Process grammar errors
            for err_type, count in grammar_stats.items():
                weight = min(0.95, 0.3 + (count / max(grammar_total, 1)) * 0.6)
                self._map_grammar_error(err_type, weight)

            # Adjust threshold based on error rate
            total_errors = spelling_total + grammar_total
            # Estimate words per review ~ 80
            total_words = total_reviews * 80
            self.error_rate = (total_errors / total_words * 100) if total_words > 0 else 0

            # Dynamic threshold: higher error rate = lower threshold
            self.threshold = 0.08 / (1 + self.error_rate / 10)

            log_with_timestamp(f"  Error rate: {self.error_rate:.2f} per 100 words")
            log_with_timestamp(f"  Dynamic threshold: {self.threshold:.2f}")

        except Exception as e:
            log_with_timestamp(f"  Warning: Could not load writing analysis: {e}")

    def _map_spelling_error(self, err_type: str, weight: float):
        """Map spelling error type to lexical patterns."""
        err_lower = err_type.lower().replace(" ", "_").replace("-", "_")

        if err_lower in self.default_patterns:
            patterns = self.default_patterns[err_lower]
            for original, error in patterns[:3]:  # Top 3 patterns
                self.lexical_map[original.lower()] = {
                    "coef": weight,
                    "action": "replace",
                    "value": error,
                    "type": err_type
                }

    def _map_grammar_error(self, err_type: str, weight: float):
        """Map grammar error type to internal weights."""
        err_lower = err_type.lower()

        if "agreement" in err_lower:
            self.grammar_weights["agreement"] = max(self.grammar_weights["agreement"], weight)
        elif "collocation" in err_lower:
            self.grammar_weights["collocation"] = max(self.grammar_weights["collocation"], weight)
        elif "preposition" in err_lower:
            self.grammar_weights["preposition"] = max(self.grammar_weights["preposition"], weight)
        elif "suffix" in err_lower:
            self.grammar_weights["suffix"] = max(self.grammar_weights["suffix"], weight)
        elif "hyphen" in err_lower:
            self.grammar_weights["hyphenation"] = max(self.grammar_weights["hyphenation"], weight)

    def calculate_risk_score(self, text: str) -> tuple:
        """Calculate risk score and identify triggers."""
        total_score = 0.0
        triggers = []
        words = text.split()

        # Lexical scanning (spelling errors)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.lexical_map:
                data = self.lexical_map[clean_word]
                total_score += data['coef']
                triggers.append({
                    "type": "spelling",
                    "subtype": data.get('type', 'lexical'),
                    "target": word,
                    "coef": data['coef'],
                    "suggestion": data['value']
                })

        # Grammar pattern scanning
        # Agreement: plural + are/do/have
        if SPACY_AVAILABLE:
            doc = nlp(text)
            for token in doc:
                if token.tag_ == "NNS":
                    if token.head.pos_ in ["VERB", "AUX"]:
                        verb = token.head.text.lower()
                        if verb in ['are', 'do', 'have', 'were']:
                            total_score += self.grammar_weights["agreement"]
                            triggers.append({
                                "type": "grammar",
                                "subtype": "agreement",
                                "target": f"{token.text} {token.head.text}",
                                "coef": self.grammar_weights["agreement"],
                                "suggestion": f"Change '{token.head.text}' to singular form"
                            })

        # Relative clause: "which/that is/are" -> remove is/are
        rel_match = re.search(r'\b(which|that)\s+(is|are)\b', text, re.IGNORECASE)
        if rel_match:
            total_score += self.grammar_weights["preposition"]
            triggers.append({
                "type": "grammar",
                "subtype": "relative_clause",
                "target": rel_match.group(0),
                "coef": self.grammar_weights["preposition"],
                "suggestion": f"Remove '{rel_match.group(2)}'"
            })

        # Hyphenation: compound words
        hyp_match = re.search(r'\b([a-z]+)-([a-z]+)\b', text, re.IGNORECASE)
        if hyp_match:
            compound = hyp_match.group(0).lower()
            if any(hw in compound for hw in ['high', 'good', 'off', 'cross', 'well']):
                total_score += self.grammar_weights["hyphenation"]
                triggers.append({
                    "type": "grammar",
                    "subtype": "hyphenation",
                    "target": hyp_match.group(0),
                    "coef": self.grammar_weights["hyphenation"],
                    "suggestion": f"Remove hyphen: '{hyp_match.group(1)} {hyp_match.group(2)}'"
                })

        return total_score, triggers

    def apply_single_error(self, text: str, triggers: list) -> str:
        """
        Apply ONLY the highest-weighted trigger (single error per query).
        This ensures natural-looking output.
        """
        if not triggers:
            return text

        # Sort by weight descending
        triggers.sort(key=lambda x: x['coef'], reverse=True)

        # Only apply if above threshold
        if triggers[0]['coef'] < self.threshold:
            return text

        trigger = triggers[0]
        modified = text

        if trigger['type'] == 'spelling':
            target = trigger['target']
            suggestion = trigger['suggestion']
            # Case-insensitive replacement
            pattern = re.compile(re.escape(target), re.IGNORECASE)
            modified = pattern.sub(suggestion, modified, count=1)

        elif trigger['type'] == 'grammar':
            subtype = trigger['subtype']
            target = trigger['target']

            if subtype == 'agreement':
                # Change are->is, do->does, have->has
                replacements = {'are': 'is', 'do': 'does', 'have': 'has', 'were': 'was'}
                for orig, repl in replacements.items():
                    if orig in target.lower():
                        modified = re.sub(r'\b' + orig + r'\b', repl, modified, count=1, flags=re.IGNORECASE)
                        break

            elif subtype == 'relative_clause':
                # Remove "is" or "are" after which/that
                modified = re.sub(r'\b(which|that)\s+(is|are)\b', r'\1', modified, count=1, flags=re.IGNORECASE)

            elif subtype == 'hyphenation':
                # Remove hyphen
                modified = modified.replace(target, target.replace('-', ' '), 1)

        return modified

    def generate_noisy_query(self, original_query: str) -> dict:
        """Generate a noisy version of the query."""
        score, triggers = self.calculate_risk_score(original_query)
        noisy_query = self.apply_single_error(original_query, triggers)

        return {
            "original": original_query,
            "noisy": noisy_query,
            "modified": original_query != noisy_query,
            "risk_score": round(score, 2),
            "triggers_found": len(triggers),
            "applied_trigger": triggers[0] if triggers and original_query != noisy_query else None
        }


def process_user(user_id: str, dual_queries_file: str, writing_analysis_file: str, output_dir: str):
    """Process a single user's queries."""
    log_with_timestamp(f"Processing user {user_id}...")

    # Load dual queries
    with open(dual_queries_file, 'r', encoding='utf-8') as f:
        dual_data = json.load(f)

    # Initialize generator with writing analysis
    generator = NoisyQueryGenerator(writing_analysis_file)

    results = []
    modified_count = 0

    # Handle both list and dict formats
    if isinstance(dual_data, list):
        queries = dual_data
    else:
        queries = dual_data.get('queries', [])

    log_with_timestamp(f"  Processing {len(queries)} queries...")

    for query_data in queries:
        asin = query_data.get('asin')
        public_query = query_data.get('public_query', '')
        personalized_query = query_data.get('personalized_query', '')

        # Generate noisy versions
        public_noisy = generator.generate_noisy_query(public_query)
        personalized_noisy = generator.generate_noisy_query(personalized_query)

        if personalized_noisy['modified']:
            modified_count += 1

        results.append({
            "asin": asin,
            "public_query": {
                "original": public_query,
                "noisy": public_noisy['noisy'],
                "modified": public_noisy['modified']
            },
            "personalized_query": {
                "original": personalized_query,
                "noisy": personalized_noisy['noisy'],
                "modified": personalized_noisy['modified'],
                "risk_score": personalized_noisy['risk_score'],
                "applied_trigger": personalized_noisy['applied_trigger']
            }
        })

    # Save results
    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "modified_queries": modified_count,
        "modification_rate": round(modified_count / len(results) * 100, 1) if results else 0,
        "error_rate_per_100_words": round(generator.error_rate, 2),
        "threshold": round(generator.threshold, 2),
        "queries": results
    }

    output_file = os.path.join(output_dir, f"noisy_queries_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    if results:
        log_with_timestamp(f"  Modified {modified_count}/{len(results)} queries ({modified_count/len(results)*100:.1f}%)")
    else:
        log_with_timestamp(f"  No queries processed")
    log_with_timestamp(f"  Saved to {output_file}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Stage 4.5: Generate Noisy Personalized Queries")
    parser.add_argument("--dual-queries-dir", required=True, help="Directory with dual_queries_*.json")
    parser.add_argument("--writing-analysis-dir", required=True, help="Directory with writing_analysis_*.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--user-ids", nargs="+", help="Specific user IDs to process (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all dual query files
    dual_files = {f.replace("dual_queries_", "").replace(".json", ""): f
                  for f in os.listdir(args.dual_queries_dir)
                  if f.startswith("dual_queries_") and f.endswith(".json")}

    # Find all writing analysis files
    writing_files = {f.replace("writing_analysis_", "").replace(".json", ""): f
                     for f in os.listdir(args.writing_analysis_dir)
                     if f.startswith("writing_analysis_") and f.endswith(".json")}

    # Get user IDs to process
    if args.user_ids:
        user_ids = args.user_ids
    else:
        user_ids = list(set(dual_files.keys()) & set(writing_files.keys()))

    log_with_timestamp(f"Found {len(user_ids)} users to process")

    summary = []
    for user_id in user_ids:
        if user_id not in dual_files:
            log_with_timestamp(f"  Skipping {user_id}: no dual queries file")
            continue
        if user_id not in writing_files:
            log_with_timestamp(f"  Skipping {user_id}: no writing analysis file")
            continue

        dual_path = os.path.join(args.dual_queries_dir, dual_files[user_id])
        writing_path = os.path.join(args.writing_analysis_dir, writing_files[user_id])

        result = process_user(user_id, dual_path, writing_path, args.output_dir)
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
            "total_users": len(summary),
            "users": summary
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nSummary saved to {summary_file}")
    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
