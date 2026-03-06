#!/usr/bin/env python3
"""
Stage 9 (V9 LLM): Single-Word Targeted Noise Injection - LLM-Based
Part of the User Profile Pipeline

This script identifies the SINGLE highest-risk word in each query
(using the spelling difficulty model) and injects a targeted spelling
error based on the user's error profile.

Key Feature: Uses LLM to inject spelling errors for more natural results.

Input:
  - Aligned queries from Stage 7 (iterative_refinement_results.json)
  - Writing analysis from Stage 4 (writing_analysis_*.json)
  - Spelling difficulty model from Stage 8

Output:
  - Queries with exactly ONE targeted spelling error per query
"""

PROMPT_VERSION = "v12_punctuation_fix"

import json
import os
import re
import sys
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from llm_client import LLMClient

# Import spelling scorer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../08_spelling_difficulty")
try:
    spelling_scorer = __import__("08_spelling_scorer")
    SpellingDifficultyScorer = spelling_scorer.SpellingDifficultyScorer
except Exception as e:
    print(f"Error loading spelling scorer: {e}")
    SpellingDifficultyScorer = None


def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# Common homophone substitutions
HOMOPHONE_MAP = {
    'there': 'their',
    'their': 'there',
    'theyre': 'there',
    'your': 'youre',
    'youre': 'your',
    'its': "it's",
    "it's": 'its',
    'whose': 'who\'s',
    'who\'s': 'whose',
    'than': 'then',
    'then': 'than',
    'loose': 'lose',
    'lose': 'loose',
    'affect': 'effect',
    'effect': 'affect',
    'accept': 'except',
    'except': 'accept',
    'advice': 'advise',
    'advise': 'advice',
    'brake': 'break',
    'break': 'brake',
    'clothes': 'cloths',
    'cloths': 'clothes',
    'could': 'could\'ve',
    'course': 'coarse',
    'coarse': 'course',
    'dear': 'deer',
    'deer': 'dear',
    'desert': 'dessert',
    'dessert': 'desert',
    'device': 'devise',
    'devise': 'device',
    'dual': 'duel',
    'duel': 'dual',
    'hear': 'here',
    'here': 'hear',
    'hole': 'whole',
    'whole': 'hole',
    'know': 'no',
    'no': 'know',
    'lead': 'led',
    'led': 'lead',
    'meat': 'meet',
    'meet': 'meat',
    'peace': 'piece',
    'piece': 'peace',
    'plain': 'plane',
    'plane': 'plain',
    'principal': 'principle',
    'principle': 'principal',
    'quiet': 'quite',
    'quite': 'quiet',
    'stationary': 'stationery',
    'stationery': 'stationary',
    'sweet': 'suite',
    'suite': 'sweet',
    'weak': 'week',
    'week': 'weak',
    'weather': 'whether',
    'whether': 'weather',
}

# Common difficult word misspellings
HARD_WORD_ERRORS = {
    'fuchsia': 'fushia',
    'rhythm': 'rythm',
    'gauge': 'guage',
    'queue': 'que',
    'yacht': 'yatch',
    'bureaucratic': 'burocratic',
    'cacophony': 'cacaphony',
    'carburetor': 'carburator',
    'catastrophe': 'catastrophy',
    'conscious': 'concious',
    'curiosity': 'curiousity',
    'definitely': 'definitly',
    'desiccate': 'desicate',
    'discrete': 'discreet',
    'disseminate': 'diseminate',
    'embarrass': 'embarass',
    'exceed': 'excede',
    'existence': 'existance',
    'fiery': 'firey',
    'fluorescent': 'florescent',
    'gauge': 'guage',
    'glamorous': 'glamorus',
    'grateful': 'greatful',
    'grievous': 'grievous',
    'hierarchy': 'hierachy',
    'humorous': 'humorus',
    'immediately': 'immediatly',
    'incidentally': 'incidently',
    'inoculate': 'innoculate',
    'irresistible': 'unresistable',
    'liaison': 'liason',
    'library': 'libary',
    'maintenance': 'maintainance',
    'maneuver': 'maneuver',
    'mischievous': 'mischievous',
    'misspell': 'mispell',
    'noticeable': 'noticable',
    'occasionally': 'ocassionally',
    'occurred': 'occured',
    'occurrence': 'occurence',
    'pavilion': 'pavillion',
    'perseverance': 'perserverance',
    'pharaoh': 'pharoah',
    'playwright': 'playwrite',
    'possession': 'posession',
    'precede': 'preceed',
    'principal': 'principle',
    'privilege': 'privelege',
    'pronunciation': 'pronounciation',
    'publicly': 'publically',
    'questionnaire': 'questionaire',
    'receive': 'recieve',
    'referred': 'refered',
    'reference': 'referance',
    'relevant': 'relevent',
    'religious': 'religous',
    'rhyme': 'rime',
    'rhythm': 'rythm',
    'sacrilegious': 'sacreligious',
    'separate': 'seperate',
    'sergeant': 'sargent',
    'supersede': 'supercede',
    'threshold': 'threshhold',
    'tomorrow': 'tommorow',
    'truly': 'truly',
    'unforeseen': 'unforseen',
    'until': 'untill',
    'unusual': 'unusual',
    'vaccine': 'vaccinne',
    'weird': 'wierd',
    'wellness': 'wellnes',
    'whether': 'wether',
    'wonderful': 'wonderfull',
}


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
        """Get spelling error profile for a user."""
        data = self.load(user_id)
        if not data:
            return {"total_errors": 0, "spelling": {}}

        stats = data.get('statistics', {})
        return {
            "total_errors": stats.get('spelling_total', 0),
            "spelling": stats.get('spelling', {}),
        }



class SingleWordInjector:
    """Injects a single spelling error into a query using LLM."""

    def __init__(self, scorer, writing_loader: WritingAnalysisLoader):
        self.scorer = scorer
        self.writing_loader = writing_loader
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            self.llm = LLMClient()
        return self.llm

    def find_target_word_and_error_type(self, query: str, user_id: str, error_profile: Dict) -> Optional[Tuple[str, str, float]]:
        """
        Find the single highest-risk word AND its best error type.

        For each vulnerable word, calculate scores for all 9 error types.
        Select the (word, error_type) combination with the highest weighted score.
        """
        if not self.scorer:
            return None

        # Get all words in query
        words = query.split()
        best_word = None
        best_error_type = None
        best_score = -1

        for word in words:
            # Skip short words
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) <= 3:
                continue

            # Get scores for all 9 error types for this word
            error_type_scores = self.scorer.predict_difficulty_by_error_type(clean_word, user_profile=error_profile)

            if not error_type_scores:
                continue

            # Find the error type with highest weighted score for this word
            for error_type, scores in error_type_scores.items():
                weighted_score = scores['weighted_score']
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_word = word
                    best_error_type = error_type

        if best_word and best_error_type:
            return (best_word, best_error_type, best_score)
        return None

    def _separate_punctuation(self, word: str) -> tuple:
        """
        Separate a word into (clean_word, leading_punct, trailing_punct).

        Examples:
            "workhorse." → ("workhorse", "", ".")
            "projects," → ("projects", "", ",")
            "'hello'" → ("hello", "'", "'")
            "applications." → ("applications", "", ".")
        """
        import re

        # Match: leading punctuation + word + trailing punctuation
        # Word can contain hyphens like "high-strength"
        pattern = r'^([\'"([{<]*)([a-zA-Z][a-zA-Z-]*)([\'"\]}>.,!?;:]*[\'"\]}>.,!?;:]*)$'
        match = re.match(pattern, word)

        if match:
            leading = match.group(1)  # e.g., "'"
            clean = match.group(2)    # e.g., "workhorse"
            trailing = match.group(3) # e.g., "."
            return (clean, leading, trailing)

        # Fallback: try to split trailing punctuation only
        pattern2 = r'^([\'"([{<]*)([a-zA-Z][a-zA-Z-]*)([\'"\]}>.,!?;:]+)$'
        match2 = re.match(pattern2, word)

        if match2:
            leading = match2.group(1)
            clean = match2.group(2)
            trailing = match2.group(3)
            return (clean, leading, trailing)

        # Last fallback: return as is
        return (word, "", "")

    def apply_error_to_word(self, word: str, error_type: str) -> str:
        """Apply a specific error pattern to a single word using LLM."""

        # CRITICAL FIX: Separate punctuation before LLM processing
        clean_word, leading_punct, trailing_punct = self._separate_punctuation(word)

        # Use clean word for LLM processing
        word_for_llm = clean_word
        error_descriptions = {
            'Deletion': {
                'operation': 'Remove exactly one letter from anywhere in the word',
                'rule': 'Choose a letter to delete (not the first/last unless word is long)',
                'examples': [
                    ("because", "becuse", "deleted 'a'"),
                    ("color", "colr", "deleted 'o'"),
                    ("interesting", "intersting", "deleted 'e'"),
                    ("construction", "contruction", "deleted 's'"),
                    ("smoothly", "smoothy", "deleted 'l'")
                ]
            },
            'Insertion': {
                'operation': 'Double one consonant (insert a duplicate)',
                'rule': 'Find a consonant and double it, typically in the middle',
                'examples': [
                    ("across", "accross", "doubled 'c'"),
                    ("until", "untill", "doubled 'l'"),
                    ("environment", "environnment", "doubled 'n'"),
                    ("strength", "strengtth", "doubled 't'"),
                    ("professional", "professsional", "doubled 's'")
                ]
            },
            'Transposition': {
                'operation': 'Swap two adjacent letters',
                'rule': 'Exchange positions of two neighboring letters',
                'examples': [
                    ("the", "teh", "swapped h-e"),
                    ("from", "form", "swapped r-o"),
                    ("believe", "belive", "swapped e-i"),
                    ("construction", "construciton", "swapped t-i"),
                    ("weight", "wieght", "swapped e-i")
                ]
            },
            'Scramble': {
                'operation': 'Rearrange 3+ letters incorrectly',
                'rule': 'Mix up multiple letters while keeping word recognizable',
                'examples': [
                    ("definitely", "definitly", "scrambled e-l-y"),
                    ("separate", "seperate", "scrambled a-r-a"),
                    ("necessary", "neccesary", "scrambled e-s-s"),
                    ("embroidery", "embrodery", "scrambled i-d-e"),
                    ("composition", "compositoin", "scrambled t-i-o")
                ]
            },
            'Substitution': {
                'operation': 'Replace one letter with a different letter',
                'rule': 'Change one letter to another that looks or sounds similar',
                'examples': [
                    ("work", "wprk", "o→p"),
                    ("great", "greAt", "a→A"),
                    ("price", "prIce", "i→I"),
                    ("bright", "briht", "g→h"),
                    ("thread", "thraed", "e→a")
                ]
            },
            'Homophone': {
                'operation': 'Replace with a word that sounds the same but different spelling',
                'rule': 'Use common homophone mistakes, phonetic misspellings, or contractions',
                'examples': [
                    ("there", "their", "homophone"),
                    ("your", "youre", "missing apostrophe"),
                    ("its", "it's", "wrong contraction"),
                    ("weight", "wait", "homophone"),
                    ("perfectly", "perfeclty", "phonetic: ly→lty"),
                    ("enough", "enuff", "phonetic: gh→ff"),
                    ("professional", "professinal", "phonetic: ion→al")
                ]
            },
            'Suffix': {
                'operation': 'Remove or modify the ending',
                'rule': 'For -ing/-ed/-s words: remove or change the ending. For other words: change last 1-2 letters.',
                'examples': [
                    ("running", "runing", "removed n"),
                    ("making", "makeing", "changed ing→e ing"),
                    ("painting", "paintin", "removed g"),
                    ("composition", "composision", "changed tion→sion"),
                    ("strength", "strengt", "removed h"),
                    ("smoothly", "smoothy", "changed ly→y"),
                    ("construction", "construcshon", "changed tion→shon")
                ]
            },
            'Hard Word': {
                'operation': 'Misspell a difficult/complex word',
                'rule': 'Make a realistic mistake in a challenging word (wrong vowels, doubled letters, etc.)',
                'examples': [
                    ("definitely", "definitly", "removed e"),
                    ("separate", "seperate", "a→e"),
                    ("necessary", "neccesary", "single s→double s"),
                    ("embossing", "embosing", "removed s"),
                    ("cardstock", "cardstok", "ck→k"),
                    ("professional", "profesional", "removed s"),
                    ("razor", "rasor", "z→s")
                ]
            },
            'Extra Space': {
                'operation': 'Split one word into two with an unnecessary space',
                'rule': 'Add one space in the middle (preferably after a common syllable break)',
                'examples': [
                    ("notebook", "note book", "space after note"),
                    ("threading", "threa ding", "space after threa"),
                    ("construction", "constr uction", "space after constr"),
                    ("background", "back ground", "space after back"),
                    ("application", "ap plication", "space after ap")
                ]
            },
        }

        error_info = error_descriptions.get(error_type, {
            'operation': 'Misspell this word',
            'rule': 'Change the spelling',
            'examples': [("test", "tets", "simple misspelling")]
        })

        # Format examples with explanations
        examples_with_notes = "\n".join([
            f"  • {orig} → {err} ({note})"
            for orig, err, note in error_info['examples'][:5]
        ])

        prompt = f"""# Spelling Error Generation Task

You need to create a specific spelling error in a target word.

## Error Type: {error_type}
**Operation**: {error_info['operation']}
**Rule**: {error_info['rule']}

## Examples (showing the pattern):
{examples_with_notes}

## Target Word: "{word_for_llm}"

## CRITICAL Rules:
1. You MUST modify letters/characters within the word
2. Do NOT only add/remove punctuation - that will be rejected
3. Do NOT only change capitalization - that will be rejected
4. Return ONLY the misspelled word (no explanation, no quotes, no "output:" prefix)

## Your Task:
Apply the "{error_type}" error pattern to "{word_for_llm}" following the rule above.

## Response (just the misspelled word):"""

        try:
            llm = self._get_llm()
            response = llm.call(prompt, max_tokens=20, temperature=0.1)

            # Strictly validate response
            response = response.strip().strip('"\'').strip().lower()

            # Remove common prefixes if present
            for prefix in ['the modified word is:', 'modified:', 'the word is:', 'output:', 'answer:']:
                if response.startswith(prefix):
                    response = response.split(':', 1)[1].strip()
                    break

            # CRITICAL: Must be a single word, different from original clean word
            if (response and
                response != clean_word.lower() and
                len(response.split()) == 1 and  # Exactly ONE word
                len(response) >= 2):  # At least 2 characters
                # Reattach punctuation to the misspelled word
                result = leading_punct + response + trailing_punct
                return result
            else:
                # Log why it failed
                log_with_timestamp(f"  LLM validation failed: got '{response}' (len={len(response.split())})")

        except Exception as e:
            log_with_timestamp(f"  LLM error: {e}")

        # Fallback: return original word with original punctuation
        return word

    def inject_single_error(self, query: str, user_id: str) -> Dict:
        """Inject exactly ONE spelling error into the query."""
        # Get user's error profile
        error_profile = self.writing_loader.get_error_profile(user_id)

        # Find target word AND best error type in one pass
        target_info = self.find_target_word_and_error_type(query, user_id, error_profile)

        if not target_info:
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": "No vulnerable word found"
            }

        target_word, error_type, weighted_score = target_info

        # Apply error to the target word using LLM
        misspelled_word = self.apply_error_to_word(target_word, error_type)

        log_with_timestamp(f"  Target: '{target_word}' → '{misspelled_word}' ({error_type})")

        # If LLM didn't modify the word, return original
        if misspelled_word.lower() == target_word.lower():
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": f"LLM could not modify '{target_word}' with {error_type}"
            }

        # Replace only the FIRST occurrence of the target word
        # Handle punctuation correctly: replace exact match including punctuation

        # Method 1: Try exact string match first (handles punctuation correctly)
        if target_word in query:
            noisy_query = query.replace(target_word, misspelled_word, 1)
        else:
            # Method 2: Use regex with word boundary (as fallback)
            # Case-insensitive replacement
            pattern = r'\b' + re.escape(target_word) + r'\b'
            noisy_query = re.sub(pattern, misspelled_word, query, count=1, flags=re.IGNORECASE)

        # Verify modification was reasonable
        # Simple check: query should be different but not completely rewritten
        orig_words = query.split()
        noisy_words = noisy_query.split()

        # Count word positions that changed
        # For Extra Space, one word becomes two, so length increases by 1
        # For other errors, length stays the same
        if abs(len(noisy_words) - len(orig_words)) > 1:
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": f"Verification failed: word count changed too much ({len(orig_words)} → {len(noisy_words)})"
            }

        # Count how many word positions are different
        min_len = min(len(orig_words), len(noisy_words))
        changed_positions = sum(1 for i in range(min_len) if orig_words[i].lower() != noisy_words[i].lower())

        # For non-Extra Space errors, only 1 word position should change
        # For Extra Space, 2 words might change (one word splits into two)
        expected_changes = 2 if error_type == 'Extra Space' else 1

        if changed_positions > expected_changes:
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": f"Verification failed: too many words changed ({changed_positions} vs expected {expected_changes})"
            }

        # Verify that change actually happened
        if noisy_query.lower() == query.lower():
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": f"Verification failed: no change detected"
            }

        return {
            "original": query,
            "noisy": noisy_query,
            "modified": True,
            "target_word": target_word,
            "misspelled_word": misspelled_word,
            "error_type": error_type,
            "weighted_score": weighted_score,
            "reason": f"Applied {error_type} to '{target_word}' (weighted_score: {weighted_score:.3f})"
        }


def process_user(user_id: str,
                 aligned_queries_file: str,
                 injector: SingleWordInjector,
                 output_dir: str):
    """Process a single user's queries."""

    log_with_timestamp(f"Processing user {user_id}...")

    # Load aligned queries from Stage 7
    with open(aligned_queries_file, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    results = []
    modified_count = 0

    # Handle different file formats
    if isinstance(aligned_data, list):
        queries = aligned_data
    elif 'queries' in aligned_data:
        queries = aligned_data['queries']
    elif 'results' in aligned_data:
        queries = aligned_data['results']
    else:
        queries = []

    log_with_timestamp(f"  Processing {len(queries)} queries...")

    for query_data in queries:
        asin = query_data.get('asin')

        # Try different field names for aligned query
        aligned_query = (
            query_data.get('final_query') or
            query_data.get('aligned_query') or
            query_data.get('final_aligned_query') or
            query_data.get('personalized_query') or
            query_data.get('query', '')
        )

        if not aligned_query:
            log_with_timestamp(f"  No aligned query found for ASIN {asin}")
            continue

        result = injector.inject_single_error(aligned_query, user_id)

        if result['modified']:
            modified_count += 1
        else:
            # Log why modification failed
            log_with_timestamp(f"  ✗ {asin}: {result.get('reason', 'Unknown reason')}")

        # Build result dict, including reason if present
        query_result = {
            "asin": asin,
            "personalized_query": {
                "original": result['original'],
                "noisy": result['noisy'],
                "modified": result['modified'],
                "method": f"single_word_{PROMPT_VERSION}"
            }
        }

        # Add reason and other details if modified
        if result['modified']:
            query_result["personalized_query"]["target_word"] = result.get("target_word")
            query_result["personalized_query"]["misspelled_word"] = result.get("misspelled_word")
            query_result["personalized_query"]["error_type"] = result.get("error_type")
            query_result["personalized_query"]["weighted_score"] = result.get("weighted_score")
            query_result["personalized_query"]["reason"] = result.get("reason")
        else:
            query_result["personalized_query"]["reason"] = result.get("reason", "Unknown reason")

        results.append(query_result)

        if result['modified']:
            log_with_timestamp(f"  ✓ {asin}: {result['reason']}")

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

    if results:
        log_with_timestamp(f"  Modified {modified_count}/{len(results)} queries ({modified_count/len(results)*100:.1f}%)")
    log_with_timestamp(f"  Saved to {output_file}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9 (V9 LLM): Single-Word Targeted Noise Injection - LLM-Based"
    )
    parser.add_argument("--stage8-results", required=True,
                        help="Path to iterative_refinement_results.json from Stage 7")
    parser.add_argument("--writing-analysis-dir", required=True,
                        help="Directory with writing_analysis_*.json from Stage 4")
    parser.add_argument("--spelling-model-path", required=True,
                        help="Path to the trained spelling difficulty model from Stage 8")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--user-ids", nargs="+",
                        help="Specific user IDs to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize writing loader
    writing_loader = WritingAnalysisLoader(args.writing_analysis_dir)

    # Initialize spelling difficulty scorer
    if not SpellingDifficultyScorer:
        log_with_timestamp("ERROR: SpellingDifficultyScorer not available")
        return 1

    log_with_timestamp("Loading spelling difficulty model...")
    scorer = SpellingDifficultyScorer(args.spelling_model_path)

    # Initialize single-word injector
    injector = SingleWordInjector(scorer, writing_loader)

    # Load Stage 7 results
    log_with_timestamp(f"Loading Stage 7 results from {args.stage8_results}...")
    with open(args.stage8_results, 'r', encoding='utf-8') as f:
        stage7_data = json.load(f)

    # Extract user queries
    # Stage 7 format: {"results": [...], "summary": {...}}
    if 'results' in stage7_data:
        all_queries = stage7_data['results']
    else:
        all_queries = stage7_data if isinstance(stage7_data, list) else []

    # Group by user_id
    user_queries = defaultdict(list)
    for q in all_queries:
        uid = q.get('user_id')
        if uid:
            user_queries[uid].append(q)

    # Filter to requested users
    if args.user_ids:
        user_ids = [u for u in args.user_ids if u in user_queries]
    else:
        user_ids = list(user_queries.keys())

    log_with_timestamp(f"Found {len(user_ids)} users to process")
    log_with_timestamp(f"Prompt version: {PROMPT_VERSION}")

    summary = []
    for user_id in user_ids:
        # Create a temporary file for this user's queries
        temp_file = f"/tmp/queries_{user_id}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(user_queries[user_id], f, indent=2)

        result = process_user(
            user_id=user_id,
            aligned_queries_file=temp_file,
            injector=injector,
            output_dir=args.output_dir
        )

        summary.append({
            "user_id": user_id,
            "total_queries": result['total_queries'],
            "modified_queries": result['modified_queries'],
            "modification_rate": result['modification_rate']
        })

        # Clean up temp file
        os.remove(temp_file)

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

    return 0


if __name__ == "__main__":
    exit(main())
