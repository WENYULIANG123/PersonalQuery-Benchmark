#!/usr/bin/env python3
"""
Amazon Review Style Analysis using Mistral-7B

This script analyzes the writing style of Amazon review users by:
1. Combining all reviews (title + text) for each user
2. Calculating linguistic features
3. Using LLM to generate style_analysis based on features and semantic context
"""

import os
import sys
import re
import time
import json
import argparse
import asyncio
from collections import Counter, defaultdict
import math
import json_repair

# Ensure stark/code is on Python path so we can import model.py
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

# Import the model from model.py
try:
    from model import get_gm_model, call_llm_with_retry, submit_batch_inference, wait_for_batch_results
except ImportError as e:
    print(f"❌ Model import failed: {e}", flush=True)
    sys.exit(1)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import cmudict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)

# Load CMU pronunciation dictionary for syllable counting
try:
    cmu_dict = cmudict.dict()
except:
    cmu_dict = {}


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================================
# Hard-coded runtime configuration (sbatch/CLI input will be ignored)
# ============================================================================
# NOTE: Per requirement, all runtime inputs are hard-coded here and any
# command-line or environment inputs will be overridden.
HARD_INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
HARD_OUTPUT_FILE = "/home/wlia0047/ar57/wenyu/result/style_analysis_AG7EF0SVBQOUX.json"
HARD_CUDA_DEVICE = 0
HARD_MAX_USERS = 1  # Only target one user
HARD_MIN_REVIEWS = 1
HARD_START_IDX = 0


# ============================================================================
# Linguistic Feature Calculation Functions
# ============================================================================

def count_syllables(word):
    """Count syllables in a word using CMU dictionary."""
    word = word.lower()
    if word in cmu_dict:
        return max([len([y for y in x if y[-1].isdigit()]) for x in cmu_dict[word]])
    # Fallback: estimate syllables (rough approximation)
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def calculate_readability_features(text):
    """Calculate readability metrics for a text."""
    if not text or len(text.strip()) == 0:
        return {}
    
    try:
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]  # Remove punctuation
        
        if len(words) == 0 or len(sentences) == 0:
            return {}
        
        # Basic statistics
        num_sentences = len(sentences)
        num_words = len(words)
        num_chars = sum(len(w) for w in words)
        
        # Average sentence length (ASL)
        asl = num_words / num_sentences if num_sentences > 0 else 0
        
        # Average syllables per word (ASW)
        total_syllables = sum(count_syllables(w) for w in words)
        asw = total_syllables / num_words if num_words > 0 else 0
        
        # Calculate readability indices
        # Flesch Reading Ease (FRE)
        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
        
        # Gunning-Fog Index
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        fog_index = 0.4 * (asl + (100 * complex_words / num_words)) if num_words > 0 else 0
        
        # Automated Readability Index (ARI)
        wps = num_words / num_sentences if num_sentences > 0 else 0
        cpw = num_chars / num_words if num_words > 0 else 0
        ari = 0.5 * wps + 4.71 * cpw - 21.43
        
        return {
            'flesch_index': round(fre, 2),
            'gunning_fog_index': round(fog_index, 2),
            'ari_index': round(ari, 2),
            'avg_sentence_length': round(asl, 2),
            'avg_syllables_per_word': round(asw, 2),
            'complex_words_ratio': round(complex_words / num_words, 3) if num_words > 0 else 0
        }
    except Exception as e:
        return {}


def calculate_lexical_diversity(text):
    """Calculate lexical diversity metrics."""
    if not text or len(text.strip()) == 0:
        return {}
    
    try:
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]
        
        if len(words) == 0:
            return {}
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Guiraud Index (GI)
        guiraud_index = unique_words / math.sqrt(total_words) if total_words > 0 else 0
        
        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'guiraud_index': round(guiraud_index, 3),
            'vocabulary_richness': round(unique_words / total_words, 3) if total_words > 0 else 0
        }
    except Exception as e:
        return {}


def calculate_grammar_features(text):
    """Calculate grammar-related features."""
    if not text or len(text.strip()) == 0:
        return {}
    
    try:
        words = word_tokenize(text)
        words_lower = [w.lower() for w in words if w.isalnum()]
        
        if len(words_lower) == 0:
            return {}
        
        # POS tagging
        pos_tags = pos_tag(words_lower)
        
        # Count POS categories
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        total_words = len(words_lower)
        
        # Calculate frequencies
        verb_freq = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                     pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                     pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_words if total_words > 0 else 0
        
        noun_freq = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                     pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_words if total_words > 0 else 0
        
        adj_freq = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                    pos_counts.get('JJS', 0)) / total_words if total_words > 0 else 0
        
        adv_freq = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                    pos_counts.get('RBS', 0)) / total_words if total_words > 0 else 0
        
        # Calculate bigrams (2-gram sequences)
        bigrams = list(zip(words_lower[:-1], words_lower[1:]))
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
        
        return {
            'verb_frequency': round(verb_freq, 3),
            'noun_frequency': round(noun_freq, 3),
            'adjective_frequency': round(adj_freq, 3),
            'adverb_frequency': round(adv_freq, 3),
            'bigram_diversity': round(bigram_diversity, 3),
            'unique_bigrams': unique_bigrams,
            'total_bigrams': total_bigrams
        }
    except Exception as e:
        return {}


def calculate_spelling_features(text):
    """Calculate basic text statistics - spelling analysis will be done by LLM."""
    if not text or len(text.strip()) == 0:
        return {}

    try:
        # Basic text statistics only - spelling analysis will be done by LLM
        words = word_tokenize(text)
        words_alpha = [w for w in words if w.isalpha()]

        if len(words_alpha) == 0:
            return {}

        # Calculate basic statistics
        total_words = len(words_alpha)
        unique_words = len(set(w.lower() for w in words_alpha))

        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': round(unique_words / total_words, 3) if total_words > 0 else 0
        }

    except Exception as e:
        return {}


def calculate_complexity_features(text):
    """Calculate complexity metrics."""
    if not text or len(text.strip()) == 0:
        return {}

    try:
        words = word_tokenize(text)
        words_lower = [w.lower() for w in words if w.isalnum()]

        if len(words_lower) == 0:
            return {}

        # POS tagging
        pos_tags = pos_tag(words_lower)

        # Count pronouns
        pronouns = ['PRP', 'PRP$', 'WP', 'WP$']
        pronoun_count = sum(1 for word, tag in pos_tags if tag in pronouns)

        # Count nouns
        nouns = ['NN', 'NNS', 'NNP', 'NNPS']
        noun_count = sum(1 for word, tag in pos_tags if tag in nouns)

        total_words = len(words_lower)

        # Pronoun density
        pronoun_density = pronoun_count / total_words if total_words > 0 else 0

        # Pronoun-Noun relationship
        pronoun_noun_ratio = pronoun_count / noun_count if noun_count > 0 else 0

        # Giveness approximation (simplified: ratio of repeated words)
        word_freq = Counter(words_lower)
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        giveness = repeated_words / len(word_freq) if len(word_freq) > 0 else 0

        return {
            'pronoun_count': pronoun_count,
            'noun_count': noun_count,
            'pronoun_density': round(pronoun_density, 3),
            'pronoun_noun_ratio': round(pronoun_noun_ratio, 3),
            'giveness': round(giveness, 3)
        }
    except Exception as e:
        return {}


def calculate_all_features(text):
    """Calculate all linguistic features for a text."""
    features = {}
    features.update(calculate_readability_features(text))
    features.update(calculate_lexical_diversity(text))
    features.update(calculate_grammar_features(text))
    features.update(calculate_complexity_features(text))
    features.update(calculate_spelling_features(text))
    return features


def format_features_for_prompt(features):
    """Format features as a readable string for prompt with detailed descriptions."""
    if not features:
        return "No features available."
    
    lines = []
    lines.append("=== READABILITY METRICS ===")
    if 'flesch_index' in features:
        lines.append(f"Flesch Reading Ease: {features['flesch_index']} (higher = easier to read, 0-100 scale)")
        lines.append(f"Gunning-Fog Index: {features['gunning_fog_index']} (lower = easier to read, typically 6-20)")
        lines.append(f"Automated Readability Index (ARI): {features['ari_index']} (lower = easier to read)")
    if 'avg_sentence_length' in features:
        lines.append(f"Average Sentence Length: {features['avg_sentence_length']} words")
    if 'avg_syllables_per_word' in features:
        lines.append(f"Average Syllables per Word: {features['avg_syllables_per_word']}")
    if 'complex_words_ratio' in features:
        lines.append(f"Complex Words Ratio: {features['complex_words_ratio']} (words with 3+ syllables)")
    
    lines.append("\n=== LEXICAL DIVERSITY ===")
    if 'guiraud_index' in features:
        lines.append(f"Guiraud Index: {features['guiraud_index']} (unique_words / sqrt(total_words), higher = more diverse)")
    if 'vocabulary_richness' in features:
        lines.append(f"Vocabulary Richness: {features['vocabulary_richness']} (unique_words / total_words, 0-1 scale)")
    if 'unique_words' in features and 'total_words' in features:
        lines.append(f"Unique Words: {features['unique_words']} out of {features['total_words']} total words")
    
    lines.append("\n=== GRAMMAR FREQUENCIES ===")
    if 'verb_frequency' in features:
        lines.append(f"Verb Frequency: {features['verb_frequency']} (verbs per word, 0-1 scale)")
        lines.append(f"Noun Frequency: {features['noun_frequency']} (nouns per word, 0-1 scale)")
        lines.append(f"Adjective Frequency: {features['adjective_frequency']} (adjectives per word, 0-1 scale)")
        lines.append(f"Adverb Frequency: {features['adverb_frequency']} (adverbs per word, 0-1 scale)")
    
    lines.append("\n=== COMPLEXITY METRICS ===")
    if 'pronoun_density' in features:
        lines.append(f"Pronoun Density: {features['pronoun_density']} (pronouns per word, 0-1 scale)")
        lines.append(f"Pronoun-Noun Ratio: {features['pronoun_noun_ratio']} (pronouns / nouns)")
        lines.append(f"Giveness: {features['giveness']} (repeated words ratio, 0-1 scale, higher = more context provided)")
    if 'pronoun_count' in features and 'noun_count' in features:
        lines.append(f"Pronoun Count: {features['pronoun_count']}, Noun Count: {features['noun_count']}")
    
    return "\n".join(lines) if lines else "No features available."


# ============================================================================
# Main Processing Functions
# ============================================================================

def combine_user_reviews(user_data):
    """Combine all reviews (title/summary + text/reviewText) for a user."""
    combined_texts = []
    for review in user_data.get('reviews', []):
        # Support both 'title'/'text' and 'summary'/'reviewText'
        title = (review.get('title') or review.get('summary') or '').strip()
        text = (review.get('text') or review.get('reviewText') or '').strip()
        if title and text:
            combined_texts.append(f"{title}. {text}")
        elif title:
            combined_texts.append(title)
        elif text:
            combined_texts.append(text)
    return " ".join(combined_texts)


def analyze_sentence_errors(sentence, sentence_idx, llm, sampling_params, candidate_words=None):
    """Analyze specific candidate words for SPELLING errors only (not grammar, punctuation, etc.).

    Args:
        sentence: The sentence to analyze
        sentence_idx: Index of the sentence
        llm: Language model
        sampling_params: Sampling parameters
        candidate_words: List of words that traditional method flagged as potential errors
    """
    if candidate_words is None:
        candidate_words = []

    candidate_str = ", ".join(f'"{word}"' for word in candidate_words) if candidate_words else "none"
    json_template = """
{
  "error_details": [
    {
      "word": "example_word",
      "error_type": "misspelled_word",
      "explanation": "brief explanation of the spelling mistake",
      "likely_correct": "correct_spelling",
      "context": "full sentence where the error appears",
      "sentence_idx": 0
    }
  ]
}"""

    system_msg = f"""You are an intelligent spelling error detector. Analyze ALL words in this candidate list and identify genuine spelling errors: {candidate_str}

INSTRUCTIONS:
- Examine each word in the candidate list above
- Consider the full sentence context: "{sentence.strip()}"
- Determine which words are REAL spelling errors that need correction

CLASSIFICATION RULES:
- ✅ CORRECT (do not report): Common words (the, a, and, but, is, are, was, were, have, has, had, do, does, did, will, would, can, could, should, may, might, must)
- ✅ CORRECT (do not report): Pronouns (I, you, he, she, it, we, they, me, my, your, his, her, our, their, this, that, these, those)
- ✅ CORRECT (do not report): Prepositions (in, on, at, to, for, of, with, by, from, into, onto, upon, about, above, below, between, among)
- ✅ CORRECT (do not report): Valid contractions (wasn't, doesn't, didn't, isn't, aren't, hasn't, haven't, won't, can't, shouldn't, couldn't, wouldn't)
- ✅ CORRECT (do not report): Recognized brand names that are intentionally misspelled (like "iPhone", "eBay") - but check for obvious typos
- ❌ ERROR: Words that are clearly misspelled, have wrong letters, missing/extra letters, or transpositions
- ❌ ERROR: Words that don't make sense in the sentence context

REPORT ONLY words that are genuinely misspelled and need correction. Be extremely critical - many words in this list may contain subtle typos or letter errors, even if they appear to be technical terms.

Respond with a JSON object containing ONLY validated spelling error details:{json_template}

STRICT REQUIREMENTS:
- Only include words that ACTUALLY APPEAR in the sentence above
- Only report words from the candidate list that are genuinely misspelled
- If a candidate word does not appear in the sentence, DO NOT include it
- If a candidate word appears in the sentence but is correctly spelled (including all contractions), DO NOT include it
- Return an empty array if no valid spelling errors are found

VALIDATION GUIDELINES:
- MANDATORY: Perform EXACT character-by-character comparison with standard spelling
- IGNORE ALL dictionary knowledge - ONLY compare letters
- RULE: If ANY letters differ from standard spelling, it IS an error
- UNIFIED STANDARD: ADDED, REMOVED, REPLACED, or REORDERED letters ALL = spelling errors
- MISSING LETTERS: Always errors (e.g., "definately" -> "definitely", "seperate" -> "separate")
- EXTRA LETTERS: Always errors (e.g., "hyalauronic" -> "hyaluronic", "moistutrizer" -> "moisturizer")
- LETTER SUBSTITUTIONS: Always errors (e.g., "recieve" -> "receive", "teh" -> "the")
- LETTER TRANSPOSITIONS: Always errors (e.g., "wierd" -> "weird")
- CRITICAL: "regimine" vs "regimen" = ERROR (missing 'e')
- CRITICAL: "moisurizers" vs "moisturizers" = ERROR (missing 't')
- Ignore punctuation-only differences
- STRICTLY check spelling of ALL words, including technical terms
- Do NOT give special treatment to technical or scientific terms - they must be spelled correctly too
- Common misspellings like "recieve"->"receive", "seperate"->"separate" should ALWAYS be flagged as errors
- FOR EACH CANDIDATE WORD: Explicitly verify if it appears in a standard English dictionary
- If a word is NOT in standard dictionary, it should be considered a spelling error
- Technical terms must also follow standard spelling conventions
- IGNORE CONTEXT INFLUENCE: Do NOT consider domain expertise or field specialization when evaluating spelling. A misspelled word is an error regardless of whether it appears in "skin care", "medical", or any other specialized context
- STRICT LETTER MATCHING: Perform mandatory character-by-character comparison against standard English dictionary spelling. ANY difference in letters (missing/extra/swapped/transposed) constitutes a spelling error, regardless of context
- DO NOT TRUST YOUR OWN DICTIONARY: Even if a word appears in your training data or internal knowledge base, perform explicit letter-by-letter verification against standard spelling
- MANDATORY LETTER-LEVEL VERIFICATION: For problematic words like "regimine", "moisurizers", "hyalauronic", "moistutrizer" - compare each character position against correct forms: "regimen", "moisturizers", "hyaluronic", "moisturizer". ANY mismatch constitutes an error, regardless of dictionary inclusion
- SPECIFIC CHECK: "regimine" (7 chars) vs "regimen" (7 chars) - position 6: 'e' vs 'n' = MISMATCH = ERROR
- SPECIFIC CHECK: "moisurizers" (11 chars) vs "moisturizers" (12 chars) - missing 't' = MISMATCH = ERROR

CHARACTER-LEVEL SPELLING AUDIT (MANDATORY):
For each candidate word, perform this exact process:
1. Identify the likely intended word (what it should be spelled as)
2. Compare character-by-character against the standard dictionary spelling
3. Flag ANY difference in letters (missing/extra/swapped/transposed)
4. Do NOT assume technical terms are exempt - check their spelling rigorously
5. Even if a word "sounds right" or is a valid brand, if letters differ from standard spelling, it is an ERROR

IMPORTANT: ALL LETTER DIFFERENCES ARE SPELLING ERRORS. Do NOT make exceptions based on whether letters are missing, extra, substituted, or transposed. Do NOT identify:
- Grammar errors (including articles a/an/the)
- Punctuation issues (exclamation marks, question marks, etc.)
- Sentence structure problems
- Style preferences
- Regional spelling variations (if they are valid alternatives)
- ANY issues where the alphabetic letters are correct but punctuation differs

CRITICAL RULE: If the only difference between the original word and your suggested correction is punctuation (like "!" or "?" or "."), DO NOT report it as a spelling error. This is a punctuation/style choice, not a spelling mistake.

FORBIDDEN PHRASES: Never use phrases like "error does not involve letter changes" or "the error does not involve actual letter changes" - ANY letter difference (missing/extra/swapped/transposed) IS a spelling error and MUST be reported.

  EXAMPLES of what TO flag:
  - "definately" -> "definitely" (missing 'i' - letter change)
  - "recieve" -> "receive" (wrong letters - e/i swapped)
  - "wierd" -> "weird" (wrong letter order - e/i swapped)
  - "seperate" -> "separate" (missing 'a' - letter missing)
  - "teh" -> "the" (wrong letters - e/h swapped)

  EXAMPLES of what NOT TO flag (punctuation/style issues):
  - "bar none!" -> "bar none" (only punctuation differs - NOT a spelling error)
  - "Good!" -> "Good" (only punctuation differs - NOT a spelling error)
  - "Hello?" -> "Hello" (only punctuation differs - NOT a spelling error)
  - "Yes." -> "Yes" (only punctuation differs - NOT a spelling error)

  REMEMBER: Spelling errors involve CHANGES TO LETTERS. Punctuation is not part of spelling."""

    prompt = f"""<s> [INST] {system_msg}

Sentence to analyze (#{sentence_idx + 1}):
```
{sentence}
```

VALIDATE these specific candidate words for spelling errors: {candidate_str}

For each candidate word, determine:
- Is this word actually misspelled (not in standard English dictionary)?
- Does the error involve actual letter changes (not just punctuation)?
- What type of spelling error is it?
- Perform character-by-character comparison with the correct spelling

MANDATORY SPELLING CHECK:
1. Break down each word into individual letters
2. Compare against the standard dictionary spelling
3. Flag any missing, extra, swapped, or transposed letters
4. Technical terms are NOT exempt - check them rigorously

IMPORTANT: Only confirm errors for words in the candidate list above. Do not find new errors.

Provide your validation results as a JSON object with "error_details" as specified.
[/INST]"""

    response_str, ok = call_llm_with_retry(llm, prompt, context="spelling_analysis")
    if not ok:
        return []

    error_details = []

    try:
        # Extract the LAST complete JSON object from LLM response (LLM may provide multiple JSON blocks)
        error_details = []

        # Find the last potential JSON block
        last_json_start = -1
        search_pos = 0
        while True:
            json_start = response_str.find('{', search_pos)
            if json_start == -1:
                break
            last_json_start = json_start
            search_pos = json_start + 1

        if last_json_start != -1:
            # Try to extract the last JSON block
            json_content = response_str[last_json_start:]
            brace_count = 0
            json_end = last_json_start

            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = last_json_start + i + 1
                        break

            if json_end > last_json_start:
                json_str = response_str[last_json_start:json_end]
                try:
                    response = json.loads(json_str, strict=False)
                    error_details = response.get('error_details', []) or []
                except json.JSONDecodeError:
                    # If the last JSON fails, try to find any valid JSON by going backwards
                    pass
    except json.JSONDecodeError:
        # If JSON parsing fails, return empty list
        error_details = []

    # 对每个错误进行精确类型分析和修正
    # 只处理真正的拼写错误，过滤掉LLM标记为有效的词
    valid_errors = []
    for error in error_details:
        if 'word' in error and 'likely_correct' in error and 'error_type' in error:
            word = error['word'].strip()
            correction = error['likely_correct'].strip()
            error_type = error['error_type'].strip()

            # 过滤掉LLM标记为有效或未知的词
            if error_type.lower() in ['valid_word', 'correct_word', 'correct_spelling', 'correctly_spelled',
                                    'no_error', 'no_spelling_error', 'valid', 'correct', 'proper_noun',
                                    'brand_name', 'technical_term', 'contraction']:
                continue  # 跳过这些非错误条目
            if correction.lower() in ['unknown', 'none', 'same', 'correct', 'no_correction', 'valid', 'n/a']:
                continue  # 跳过没有明确纠正的条目

            # 使用精确的错误类型分析函数
            precise_error_type = analyze_spelling_error_type(word, correction)

            # 更新错误类型为精确的中文描述
            error['error_type'] = precise_error_type

            # 添加更详细的解释
            if precise_error_type == "增减字母错误 (增加字母)":
                error['explanation'] = f"单词 '{word}' 多了一个字母，应该是 '{correction}'"
            elif precise_error_type == "增减字母错误 (减少字母)":
                error['explanation'] = f"单词 '{word}' 少了一个字母，应该是 '{correction}'"
            elif precise_error_type == "字母顺序错误":
                error['explanation'] = f"单词 '{word}' 中有两个字母顺序颠倒，应该是 '{correction}'"
            elif precise_error_type == "遗漏字母错误":
                error['explanation'] = f"单词 '{word}' 遗漏了一个字母，应该是 '{correction}'"
            elif precise_error_type == "重复字母错误":
                error['explanation'] = f"单词 '{word}' 中某个字母重复了，应该是 '{correction}'"

            # 为真正的拼写错误生成详细的LLM解释
            try:
                explanation_prompt = f"""Analyze this spelling error in detail:

Incorrect word: "{word}"
Correct word: "{correction}"
Context: "{sentence.strip()}"

Please provide a detailed explanation of:
1. Why "{word}" is incorrect
2. Why "{correction}" is the correct spelling
3. The linguistic or etymological reason for the correct form
4. Any common patterns or rules this illustrates

Be specific and educational in your explanation."""

                detailed_explanation, ok = call_llm_with_retry(llm, explanation_prompt, context="spelling_explanation")
                if not ok:
                    detailed_explanation = "Failed to generate detailed explanation."
                error['detailed_llm_explanation'] = detailed_explanation

                print(f"  📖 LLM Explanation for '{word}' → '{correction}':")
                print(f"     {detailed_explanation[:200]}{'...' if len(detailed_explanation) > 200 else ''}")

            except Exception as e:
                print(f"  ⚠️  Failed to generate detailed explanation for '{word}': {e}")
                error['detailed_llm_explanation'] = f"Failed to generate explanation: {e}"

            valid_errors.append(error)

    # 只保留验证为真正的错误
    error_details = valid_errors

    return error_details


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def analyze_spelling_error_type(word, correction):
    """
    精确分析拼写错误类型, 返回用户指定的四种类型之一.

    Args:
        word: 错误单词 (来自文本)
        correction: 正确的单词 (来自词典/LLM建议)

    Returns:
        str: 错误类型 ("增减字母错误", "字母顺序错误", "遗漏字母错误", "重复字母错误", 或 "其他错误")
    """
    if not correction or word == correction:
        return "其他错误"

    word = word.lower()
    correction = correction.lower()

    len_diff = len(correction) - len(word)

    # 1. 字母顺序错误 (transposition) - 长度相同，有两个相邻字符交换
    if len(word) == len(correction):
        # 检查是否有两个相邻字符交换位置
        for i in range(len(word) - 1):
            if word[i] == correction[i+1] and word[i+1] == correction[i]:
                # 检查其他字符是否相同
                temp_word = list(word)
                temp_word[i], temp_word[i+1] = temp_word[i+1], temp_word[i]
                if ''.join(temp_word) == correction:
                    return "字母顺序错误"

        # 检查重复字母错误 (如 definately → definitely)
        word_counts = {}
        correction_counts = {}

        for char in word:
            word_counts[char] = word_counts.get(char, 0) + 1
        for char in correction:
            correction_counts[char] = correction_counts.get(char, 0) + 1

        # 如果错误单词中某个字母出现次数更多，可能是重复字母错误
        repeated_chars = []
        for char in word_counts:
            if word_counts.get(char, 0) > correction_counts.get(char, 0):
                repeated_chars.append(char)

        if repeated_chars and len(repeated_chars) == 1:
            return "重复字母错误"

    # 2. 遗漏字母错误 - 错误单词比正确单词短的情况
    if len_diff > 0:  # word比correction短，说明word遗漏了字母
        return "遗漏字母错误"

    # 3. 增减字母错误 - 错误单词比正确单词长的情况
    if len_diff < 0:  # word比correction长，说明word增加了字母
        return "增减字母错误"

    # 4. 其他情况
    return "其他错误"

def traditional_spell_check_sentence(sentence, spell_checker):
    """Simplified: Extract all words and pass them to LLM for judgment.
    No traditional filtering - let LLM decide what constitutes a spelling error.
    """
    # Extract all words from the sentence (improved tokenization for contractions)
    # This regex handles contractions like "wasn't", "don't", "it's" properly
    words = re.findall(r'\b\w+(?:\'\w+)*\b', sentence.lower())

    # Print tokenization result for debugging
    print(f"  🔤 Tokenized sentence: {sentence.strip()}")
    print(f"  📝 Words found: {words}")
    print(f"  🔢 Total words: {len(words)}")

    candidates = []
    for word in words:
        # Skip only the most obvious non-text elements
        if re.match(r'^\d+$', word):  # Skip pure numbers
            continue

        # Create candidate for EVERY word - let LLM judge
        candidates.append({
            'word': word,
            'sentence': sentence.strip(),
            'traditional_corrections': [],  # No traditional corrections needed
            'best_match': word,  # Default to original word
            'confidence': 'unknown'  # Let LLM determine confidence
        })

    return candidates

def load_spell_dictionary():
    """Load spell checker using pyspellchecker library."""
    try:
        from spellchecker import SpellChecker

        # Initialize spell checker with English
        spell = SpellChecker(language='en')

        # Add domain-specific words for beauty/skincare
        domain_words = {
            'hyaluronic', 'moisturizer', 'cleanser', 'toner', 'exfoliate', 'hydrate',
            'regimen', 'bristle', 'soniccare', 'phillips', 'irysa', 'fyi', 'regimine',
            'hyalauronic', 'moistutrizer', 'definately', 'seperate', 'occured', 'recieve',
            'wierd', 'accomodate', 'begining', 'commited', 'exaggerate', 'occassion',
            'priviledge', 'reccommend', 'tommorow', 'untill'
        }

        # Add words to spell checker
        for word in domain_words:
            spell.word_frequency.load_words([word])

        return spell, {}

    except ImportError:
        # Fallback: return None and use basic implementation
        print("Warning: pyspellchecker not available, using fallback method")
        return None, {}

def generate_spelling_analysis(text, llm, sampling_params):
    """Generate spelling error analysis using hybrid approach: traditional filtering + LLM verification."""
    # Load spell checker
    spell_checker, known_corrections = load_spell_dictionary()

    # Split text into sentences
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except ImportError:
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

    print(f"  🔍 Phase 1: Traditional spell checking on {len(sentences)} sentences...")

    # Phase 1: Traditional spell checking to identify candidates
    all_candidates = []
    for idx, sentence in enumerate(sentences):
        if len(sentence.strip()) < 5:  # Skip very short sentences
            continue

        sentence_candidates = traditional_spell_check_sentence(sentence, spell_checker)
        for candidate in sentence_candidates:
            candidate['sentence_idx'] = idx
        all_candidates.extend(sentence_candidates)

    print(f"  📊 Found {len(all_candidates)} potential spelling errors via traditional method")

    # Phase 2: LLM verification of candidates
    print(f"  🤖 Phase 2: LLM verification of {len(all_candidates)} candidates...")

    verified_errors = []
    processed_count = 0

    # Group candidates by sentence for batch processing
    sentence_groups = {}
    for candidate in all_candidates:
        sent_idx = candidate['sentence_idx']
        if sent_idx not in sentence_groups:
            sentence_groups[sent_idx] = []
        sentence_groups[sent_idx].append(candidate)

    # Process each sentence with candidates
    for sent_idx, candidates in sentence_groups.items():
        sentence = sentences[sent_idx]
        processed_count += 1

        print(f"    📝 Verifying sentence {processed_count}/{len(sentence_groups)} (#{sent_idx + 1}) - {len(candidates)} candidates...")

        try:
            # Get LLM verification for this sentence - only validate the traditional candidates
            candidate_words_for_sentence = [c['word'] for c in candidates]
            sentence_errors = analyze_sentence_errors(sentence, sent_idx, llm, sampling_params, candidate_words_for_sentence)

            # Cross-reference with traditional candidates
            for error in sentence_errors:
                error_word = error.get('word', '').lower()

                # Find matching traditional candidate
                matching_candidate = None
                for candidate in candidates:
                    if candidate['word'].lower() == error_word:
                        matching_candidate = candidate
                        break

                if matching_candidate:
                    # Enhance error with traditional suggestions
                    error['traditional_corrections'] = matching_candidate['traditional_corrections']
                    # Calculate edit distance between error word and correction
                    if 'likely_correct' in error:
                        error_word = error.get('word', '')
                        correction_word = error['likely_correct']
                        error['edit_distance'] = levenshtein_distance(error_word, correction_word)
                    else:
                        error['edit_distance'] = 0

                verified_errors.append(error)

            if sentence_errors:
                print(f"      ✅ LLM confirmed {len(sentence_errors)} spelling error(s)")
            else:
                print(f"      ❌ LLM rejected all {len(candidates)} traditional candidates")

        except Exception as e:
            print(f"      ⚠️  Warning: Failed to verify sentence {sent_idx + 1}: {e}")
            continue

    print(f"  ✅ Completed hybrid analysis")
    print(f"  📈 Traditional method found {len(all_candidates)} candidates, LLM verified {len(verified_errors)} as actual errors")

    # Generate overall spelling analysis summary
    error_count = len(verified_errors)

    if error_count == 0:
        spelling_errors = f"The text contains no clear spelling errors. Analyzed {len(sentences)} sentences using hybrid traditional+LLM approach and found no mistakes."
    else:
        error_types = {}
        for error in verified_errors:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Create summary description
        type_summary = ", ".join([f"{count} {error_type}" for error_type, count in error_types.items()])
        spelling_errors = f"The text contains {error_count} verified spelling errors across {len(sentences)} analyzed sentences. Error types include: {type_summary}. Analysis used hybrid approach: traditional spell checking for candidate identification, followed by LLM verification for accuracy."

    return {
        'spelling_errors': spelling_errors,
        'error_details': verified_errors
    }


def generate_style_analysis(text, features_str, llm, sampling_params):
    """Generate personalized style_analysis using LLM based on actual text content."""
    system_msg = """You are a linguistic style analyst. Your task is to analyze a user's writing style based primarily on their actual text content, using the provided linguistic features only as optional background context. Provide a detailed, personalized description that captures how this person naturally speaks and writes. The goal is to create a description that would allow someone to write a query in this user's authentic voice.

Respond with a JSON object containing two keys:
{
"style_analysis": A comprehensive, detailed, and personalized description of the user's speaking/writing style. You MUST analyze the actual text content and cover ALL 10 dimensions listed below in detail. Each dimension should have at least 2-3 sentences of analysis. IMPORTANT: Do NOT cite or attribute observations to specific numeric feature values. Describe patterns semantically and naturally (avoid phrases like "because the Flesch score is X" or inserting numeric values). You may use the provided features internally to inform the analysis, but the output should not include numeric references. Aim for 2500-3500 characters in the narrative.

  "concise_profile": A short, single-paragraph semantic profile (100-200 words) that summarizes the user's typical vocabulary, tone, sentence structure, emotional stance, and primary focus areas. The concise profile should be written as an actionable descriptor useful for quick reference or persona prompts. It must be between 100 and 200 words and presented as plain natural language (no headings, lists, or JSON inside this field).
}

You MUST provide detailed analysis for ALL of these 10 dimensions (write as a cohesive narrative, but ensure each dimension is thoroughly covered):
1. Vocabulary choices: What words does this person prefer? Simple or complex? Technical or everyday? Any favorite phrases or expressions? Any slang, abbreviations, or casual language? What specific words or phrases do they frequently use?
2. Sentence structure: How does this person construct sentences? Short and direct, or long and complex? Do they use questions, exclamations, or mostly statements? How do they connect ideas? What sentence patterns are common?
3. Emotional expression: How does this person express feelings? Directly or indirectly? Strongly or mildly? What emotional vocabulary do they use? Do emotions vary or stay consistent? What specific emotional words or phrases appear frequently?
4. Formality and tone: Is their language formal or casual? Polite or direct? How do they address the reader? Do they use "I" frequently or stay objective? What is their overall communication tone?
5. Information organization: How do they structure their thoughts? Do they use logical connectors? Is information dense or spread out? How do they transition between topics? What organizational patterns do you observe?
6. Expression habits: Any repeated phrases or patterns? How do they emphasize points? Do they use examples, comparisons, or other rhetorical devices? What specific rhetorical techniques do they employ?
7. Interaction style: Do they seem aware of the reader? Do they ask questions, give advice, or share experiences? How much do they use first-person vs. third-person? What is their level of reader engagement?
8. Domain knowledge: Do they use technical terms? How deep is their product knowledge? Do they describe technical details or focus on usage? What specific domain expertise do they demonstrate?
9. Time and space: How do they reference time (specific dates, vague terms)? How do they describe locations or spatial relationships? What verb tenses do they prefer? What temporal and spatial patterns emerge?
10. Evaluation patterns: How do they structure their reviews/opinions? What aspects do they focus on (function, appearance, price, etc.)? How do they provide recommendations? What is their typical review structure?

IMPORTANT:
- Cover ALL 10 dimensions with detailed analysis (2-3 sentences minimum per dimension)
- When you observe patterns in the text, describe them semantically without referencing specific numeric values
- Aim for a comprehensive description of 2500-3500 characters for style_analysis
- Write as if you're describing a person's unique voice and communication style, focusing on natural-language observations and examples rather than numeric evidence
- Provide specific textual examples from the user's content when possible}"""
    
    # Use more text content (up to 5000 chars) to get better style analysis
    text_sample = text[:5000] if len(text) > 5000 else text
    
    prompt = f"""<s> [INST] {system_msg}

Linguistic Features (use these numerical values as supporting evidence):
{features_str}

User's Actual Text Content:
```
{text_sample}
```

Analyze this user's writing style by examining BOTH the actual text content above AND the linguistic features provided. Use the numerical features internally to inform your interpretation, but do NOT include numeric values or explicit numeric references in the `style_analysis` output. Describe causes and patterns semantically and naturally (for example say "tends to use concise sentences" rather than "average sentence length is X").

Describe their natural speaking/writing patterns in detail, focusing on how they express themselves, what words they choose, how they structure their thoughts, and their communication habits. The goal is to create a description that captures their authentic voice so someone could write a query that sounds like it came from this user.

For the spelling_errors field, examine the text for spelling patterns, accuracy, and any systematic errors. Describe whether they write with careful attention to spelling or if they have particular tendencies in spelling mistakes.

Provide your analysis as a JSON object with "style_analysis" and "concise_profile" fields as specified above.[/INST]"""
    
    response_str, ok = call_llm_with_retry(llm, prompt, context="style_analysis")
    if not ok:
        return {'style_analysis': '', 'concise_profile': ''}

    style_analysis = ""
    concise_profile = ""

    try:
        response = json.loads(response_str, strict=False)
        # Support both keys if model returned them
        style_analysis = response.get('style_analysis', '') or response.get('styleAnalysis', '') or ""
        concise_profile = response.get('concise_profile', '') or response.get('conciseProfile', '') or response.get('concise_profile', '') or ""
    except json.JSONDecodeError:
        # Try to extract fields from text format
        style_match = re.search(r'style_analysis["\']?\s*:\s*["\']?(.*?)(?:\n\s*[}"\']|$)', response_str, re.DOTALL | re.IGNORECASE)
        if style_match:
            style_analysis = style_match.group(1).strip().strip('"\'' )

        concise_match = re.search(r'concise_profile["\']?\s*:\s*["\']?(.*?)(?:\n\s*[}"\']|$)', response_str, re.DOTALL | re.IGNORECASE)
        if concise_match:
            concise_profile = concise_match.group(1).strip().strip('"\'' )

        # If still can't extract anything, fall back to whole response as style_analysis
        if not style_analysis and not concise_profile:
            style_analysis = response_str

    return {
        'style_analysis': style_analysis,
        'concise_profile': concise_profile
    }


def generate_concise_profile(text, features_str, llm):
    """Generate a concise semantic profile (100-200 words) summarizing the user's style.
    The profile should be a short natural-language paragraph useful as a quick descriptor."""
    # Use a smaller max_tokens for a short summary
    sampling_params_short = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=300)

    # Limit text sample to reasonable size to keep prompt short
    text_sample = text[:3000] if len(text) > 3000 else text

    prompt = (
        "You are a concise linguistic profiler. Produce one single-paragraph semantic profile "
        "(100-200 words) that summarizes the user's typical vocabulary, tone, sentence structure, "
        "emotional stance, and primary focus areas. Use the numeric features below as supporting "
        "evidence but keep the profile short and actionable.\n\n"
        "Linguistic Features:\n"
        f"{features_str}\n\n"
        "User text sample:\n"
        f"{text_sample}\n\n"
        "Write a single natural-language paragraph (100-200 words). Do NOT include headings, "
        "lists, or JSON — just the paragraph."
    )

    response_str, ok = call_llm_with_retry(llm, prompt, context="concise_profile")
    if not ok:
        return ""
    return response_str


def process_users(input_file, output_file, llm, sampling_params, max_users=None, start_idx=0, min_reviews=1):
    """Process users and generate style analysis."""
    print(f"Loading data from: {input_file}")
    users = None
    # Support multiple input formats:
    # 1) Existing users JSON: {"users": { user_id: {review_count, reviews:[{title,text},...]}}}
    # 2) A JSONL file with one review per line (common Amazon format)
    # 3) A JSON array or file of review objects (list of reviews)
    with open(input_file, 'r', encoding='utf-8') as f:
        # Heuristic: treat .jsonl as line-delimited JSON
        if input_file.lower().endswith('.jsonl'):
            users = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_id = (obj.get('reviewerID') or obj.get('reviewerId') or obj.get('userId') or
                           obj.get('user_id') or obj.get('user') or obj.get('userid') or
                           obj.get('author') or obj.get('author_id'))
                title = obj.get('summary') or obj.get('title') or ""
                text = obj.get('reviewText') or obj.get('text') or obj.get('review_body') or ""
                # Attempt to capture Amazon item identifier (ASIN / Axxx field). Support common key variants.
                asin = (obj.get('asin') or obj.get('ASIN') or obj.get('asinId') or
                        obj.get('asin_id') or obj.get('product_id') or obj.get('item_id') or
                        obj.get('productId') or obj.get('itemId'))
                if user_id is None:
                    continue
                if user_id not in users:
                    users[user_id] = {"review_count": 0, "reviews": []}
                users[user_id]["reviews"].append({"title": title, "text": text, "asin": asin})
                users[user_id]["review_count"] += 1
        else:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fall back: try reading line-by-line (in case file is actually jsonl but lacks .jsonl suffix)
                f.seek(0)
                users = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    user_id = (obj.get('reviewerID') or obj.get('reviewerId') or obj.get('userId') or
                               obj.get('user_id') or obj.get('user') or obj.get('userid') or
                               obj.get('author') or obj.get('author_id'))
                    title = obj.get('summary') or obj.get('title') or ""
                    text = obj.get('reviewText') or obj.get('text') or obj.get('review_body') or ""
                    asin = (obj.get('asin') or obj.get('ASIN') or obj.get('asinId') or
                            obj.get('asin_id') or obj.get('product_id') or obj.get('item_id') or
                            obj.get('productId') or obj.get('itemId'))
                    if user_id is None:
                        continue
                    if user_id not in users:
                        users[user_id] = {"review_count": 0, "reviews": []}
                    users[user_id]["reviews"].append({"title": title, "text": text, "asin": asin})
                    users[user_id]["review_count"] += 1
            else:
                # If loaded JSON has top-level "users", use it. Otherwise, if it's a list of reviews, convert it.
                if isinstance(data, dict) and 'users' in data:
                    users = data['users']
                elif isinstance(data, list):
                    users = {}
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        user_id = (obj.get('reviewerID') or obj.get('reviewerId') or obj.get('userId') or
                                   obj.get('user_id') or obj.get('user') or obj.get('userid') or
                                   obj.get('author') or obj.get('author_id'))
                        title = obj.get('summary') or obj.get('title') or ""
                        text = obj.get('reviewText') or obj.get('text') or obj.get('review_body') or ""
                        asin = (obj.get('asin') or obj.get('ASIN') or obj.get('asinId') or
                                obj.get('asin_id') or obj.get('product_id') or obj.get('item_id') or
                                obj.get('productId') or obj.get('itemId'))
                        if user_id is None:
                            continue
                        if user_id not in users:
                            users[user_id] = {"review_count": 0, "reviews": []}
                        users[user_id]["reviews"].append({"title": title, "text": text, "asin": asin})
                        users[user_id]["review_count"] += 1
                else:
                    # Unexpected structure: try to extract 'users' key if present
                    if isinstance(data, dict) and any(k.lower() == 'users' for k in data.keys()):
                        users = data.get('users', {})
                    else:
                        # As a last resort, set users to empty dict
                        users = {}

async def process_users_batch(input_file, output_file, max_users=None, start_idx=0, min_reviews=5):
    """Process multiple users in batch mode using SiliconFlow Batch API."""
    print(f"\n🔄 STARTING BATCH PROCESSING (input: {input_file})")
    
    # Load users
    users = {}
    if input_file.endswith('.gz'):
        import gzip
        try:
            print(f"📖 Opening gzip file: {input_file}")
            with gzip.open(input_file, 'rt', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line_idx % 100000 == 0 and line_idx > 0:
                        print(f"   Processed {line_idx} lines...")
                    if not line.strip(): continue
                    try:
                        review = json.loads(line)
                        rev_user_id = review.get('reviewerID')
                        if rev_user_id:
                            if rev_user_id not in users:
                                users[rev_user_id] = {'review_count': 0, 'reviews': []}
                            users[rev_user_id]['reviews'].append(review)
                            users[rev_user_id]['review_count'] += 1
                    except Exception: continue
            print(f"✅ Finished reading gzip. Total users: {len(users)}")
        except Exception as e:
            print(f"❌ Error reading gzip file: {e}")
            return
    else:
        # Assume it's the preprocessed JSON with nested structure
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if 'results' in data and isinstance(data['results'], list):
                    users = {u.get('user_id', f"unknown_{i}"): u for i, u in enumerate(data['results'])}
                elif 'users' in data:
                    users = data['users']
                else:
                    users = data

    total_users_count = len(users)
    print(f"Total users found: {total_users_count}")
    
    # Filter and slice
    filtered_users = []
    for uid, udata in users.items():
        # Specifically target the user if we are in HARD mode or look for reviewerID
        rc = udata.get('review_count', len(udata.get('reviews', [])))
       # Create prompts for each review individually
    prompts = []
    user_metadata = {} # Map req-i -> {user_id, type, review_id}
    
    # Isolate target user or filtered users
    target_user_id = "AG7EF0SVBQOUX"
    if target_user_id in users:
        print(f"🎯 Target user {target_user_id} found in data with {users[target_user_id]['review_count']} reviews.")
        users_to_process = [(target_user_id, users[target_user_id])]
    else:
        # ... logic for other users if needed ...
        users_to_process = [] 

    # For this specific task, we only process the target user's reviews individually
    for user_id, user_data in users_to_process:
        reviews = user_data.get('reviews', [])
        print(f"Processing {len(reviews)} reviews for user {user_id}...")
        
        for idx, review in enumerate(reviews):
            title = (review.get('title') or review.get('summary') or '').strip()
            text = (review.get('text') or review.get('reviewText') or '').strip()
            full_text = f"{title}. {text}" if title and text else (title or text)
            
            if not full_text: continue
            
            spelling_prompt = f"""You are a spelling and grammar error detector. 
Analyze the following text for spelling and grammatical errors. 

TEXT:
"{full_text}"

ERROR CATEGORIES (Two Dimensions):

**DIMENSION 1: 字符与单词变体习惯 (Character & Word Variants)**
1. Deletion: Missing character, especially in long words or foreign words (e.g., "Appliqu" → "Applique")
2. Insertion: Extra character, often in brand names or repeated letters (e.g., "obssessed" → "obsessed")
3. Transposition: Adjacent letters swapped due to typing (e.g., "invididual" → "individual")
5. Substitution: Wrong character, often in brand names or similar sounds (e.g., "Kolors" → "Colors")
6. Homophone: Sound-alike words with different meanings (e.g., "pallet" → "palette", "lay" → "lie")
8. Hard Word: Misspelling of rare/technical/brand terms (e.g., "Vandycke" → "Vandyke")
9. Extra Space: Compound words incorrectly split (e.g., "paint brush" → "paintbrush")
10. Extra Hyphen: Unnecessary hyphen in phrases (e.g., "oh-so-pretty" → "oh so pretty")

**DIMENSION 2: 语法与结构逻辑习惯 (Grammar & Structural Logic)**
7. Suffix: Wrong word form, adjective/adverb confusion (e.g., "embroidered great" → "greatly")
11. Agreement: Number disagreement between subject-verb or pronoun-noun (e.g., "these kit" → "these kits", "was" → "were")
12. Hyphenation: Missing hyphen in compound adjectives before nouns (e.g., "high end machine" → "high-end machine")
13. Pronoun: Incorrect relative pronoun in clauses (e.g., "what I consider" → "which I consider")
14. Collocation: Unnatural verb-noun pairing (e.g., "brush can write" → "brush can paint/draw")
15. Preposition: Missing preposition in complex phrases (e.g., "too many of other" → "too many others")

INSTRUCTIONS:
Identify any spelling or grammatical errors in the text.
For each error, classify it into ONE of the above categories (1-15).
Be LIBERAL in detection - it's better to flag potential errors that will be validated later.
Strictly output JSON.
Format: {{"error_details": [{{"word": "wrong", "likely_correct": "right", "explanation": "reason", "category_id": 1, "category_name": "Deletion", "dimension": 1}}, ...]}}
Do NOT use tuples. Use OBJECTS with curly braces {{}}.
If no errors are found, return {{"error_details": []}}.
"""
            req_id = f"req-{len(prompts)}"
            prompts.append(spelling_prompt)
            user_metadata[req_id] = {
                "user_id": user_id, 
                "review_idx": idx, 
                "type": "spelling",
                "text_preview": full_text[:50],
                "original_text": full_text  # Store full text for Stage 2 validation
            }

    if not prompts:
        print("No prompts to submit.")
        return

    # 2. Submit Batch
    print(f"🚀 Submitting batch for {len(prompts)} prompts...")
    batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
    print(f"⏳ Batch submitted! ID: {batch_id}. Waiting for results...")
    
    # 3. Wait for Stage 1 Results
    print("⏳ Stage 1: Waiting for initial error detection...")
    results_raw = await wait_for_batch_results(batch_id, poll_interval=30)
    
    # 4. Process Stage 1 Results
    print("📊 Stage 1: Processing detected errors...")
    user_final = defaultdict(dict)
    stage1_errors = []  # Collect all errors for Stage 2 validation
    
    for res in results_raw:
        custom_id = res.get('custom_id', '')
        content = res.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
        
        meta = user_metadata.get(custom_id, {})
        user_id = meta.get('user_id', 'unknown')
        
        # Extract JSON from content
        llm_json = {}
        try:
            llm_json = json_repair.loads(content)
        except Exception:
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    llm_json = json_repair.loads(json_match.group(0))
            except Exception as e:
                print(f"Error parsing JSON for {custom_id}: {e}") 
        
        if meta['type'] == 'spelling':
            # Handle both formats
            if isinstance(llm_json, list):
                errors = llm_json
            elif isinstance(llm_json, dict):
                errors = llm_json.get('error_details', [])
            else:
                errors = []
            
            if errors:
                for err in errors:
                    # Robustness: ensure essential field 'word' exists
                    if not err.get('word'):
                        continue
                    err['review_idx'] = meta['review_idx']
                    err['context_snippet'] = meta['text_preview']
                    err['user_id'] = user_id
                    err['original_text'] = meta.get('original_text', '')
                    stage1_errors.append(err)
    
    print(f"✅ Stage 1 complete: Detected {len(stage1_errors)} potential errors.")
    
    # ========== STAGE 2: VALIDATION ==========
    if not stage1_errors:
        print("No errors detected in Stage 1. Skipping Stage 2.")
    else:
        print(f"\n🔍 Stage 2: Validating {len(stage1_errors)} detected errors...")
        
        # Prepare Stage 2 validation prompts
        validation_prompts = []
        validation_metadata = {}
        
        for idx, err in enumerate(stage1_errors):
            validation_prompt = f"""You are a spelling and grammar error validator. 
A potential error has been detected in an Amazon product review. Your job is to determine if this should be KEPT (as a true error) or REJECTED (as a stylistic choice or acceptable informal usage).

ORIGINAL TEXT:
"{err.get('original_text', '')}"

DETECTED ERROR:
- Word: "{err.get('word', 'Unknown')}"
- Suggested Correction: "{err.get('likely_correct', 'N/A')}"
- Reason: {err.get('explanation', 'None provided')}
- Category: {err.get('category_name', 'Unknown')}

VALIDATION CRITERIA (REJECT conditions):
REJECT this error if any of the following apply:
1. **Informal Punctuation**: The "error" is just a stylistic punctuation choice (e.g., "!." at the end of a sentence).
2. **Acceptable Hyphenation**: Missing hyphen in compound adjectives (e.g., "high quality") where meaning is clear.
3. **Stylistic Fragments**: Sentence fragments that work in context (e.g., "Beautiful to look at.").
4. **Casing & Brands**: Lowercase brand names (e.g., "mod podge") or language names (e.g., "english") that are otherwise correct.
5. **Idioms & Colloquialisms (New)**: Natural idiomatic expressions or metaphors (e.g., "cannot touch this many designs" meaning incomparable, "go back to shape").
6. **References & Abbreviations (New)**: Specific technical abbreviations or site identifiers (e.g., "Singerco" referring to the company/URL, "pdf" used as a generic noun).
7. **Technical/Brand Terms**: Specific product names or niche technical terms (e.g., "Sennelier", "Inktense").
8. **Accepted Variants**: Standard spelling variants (e.g., "color" vs "colour").

Otherwise, if it's a clear typo (e.g., "invididua", "obssessed") or serious grammar failure, KEEP the error.

INSTRUCTIONS:
Output JSON with your decision.
Format: {{"is_valid_error": true/false, "confidence": 0.0-1.0, "reason": "concise explanation"}}
Set "is_valid_error" to FALSE if it falls into the REJECT categories above.
"""
            req_id = f"val-{idx}"
            validation_prompts.append(validation_prompt)
            validation_metadata[req_id] = {"error_index": idx}
        
        # Submit Stage 2 batch
        print(f"🚀 Submitting Stage 2 validation batch for {len(validation_prompts)} errors...")
        # Collect custom IDs
        validation_custom_ids = list(validation_metadata.keys())
        validation_batch_id = submit_batch_inference(validation_prompts, model="Qwen/QwQ-32B", custom_ids=validation_custom_ids)
        print(f"⏳ Stage 2 batch submitted! ID: {validation_batch_id}")
        
        # Wait for Stage 2 results
        validation_results_raw = await wait_for_batch_results(validation_batch_id, poll_interval=30)
        
        # Process Stage 2 validation results
        print("📊 Stage 2: Processing validation results...")
        validated_errors = []
        
        for res in validation_results_raw:
            custom_id = res.get('custom_id', '')
            content = res.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            
            meta = validation_metadata.get(custom_id, {})
            error_idx = meta.get('error_index', -1)
            
            if error_idx == -1:
                continue
            
            # Parse validation result
            validation_json = {}
            try:
                validation_json = json_repair.loads(content)
            except Exception:
                try:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        validation_json = json_repair.loads(json_match.group(0))
                except Exception as e:
                    print(f"Error parsing validation JSON for {custom_id}: {e}")
            
            # Check if error is validated
            if isinstance(validation_json, list) and len(validation_json) > 0:
                validation_json = validation_json[0]
                
            if isinstance(validation_json, dict):
                is_valid = validation_json.get('is_valid_error', False)
                confidence = validation_json.get('confidence', 0.0)
                reason = validation_json.get('reason', '')
            else:
                is_valid = False
                confidence = 0.0
                reason = f"Invalid validation format: {type(validation_json)}"
            
            # Lower threshold and add debug info
            if is_valid and confidence >= 0.3:  # Lowered from 0.5 to 0.3
                original_error = stage1_errors[error_idx]
                original_error['validation_confidence'] = confidence
                original_error['validation_reason'] = reason
                validated_errors.append(original_error)
            else:
                # Debug: log rejected errors
                if error_idx < len(stage1_errors):
                    rejected_err = stage1_errors[error_idx]
                    print(f"  [REJECTED] {rejected_err.get('word')} -> {rejected_err.get('likely_correct')} (confidence: {confidence}, reason: {reason[:50]}...)")
        
        print(f"✅ Stage 2 complete: {len(validated_errors)}/{len(stage1_errors)} errors validated.")
        
        # Aggregate validated errors by user
        for err in validated_errors:
            user_id = err['user_id']
            if 'spelling_errors' not in user_final[user_id]:
                user_final[user_id]['spelling_errors'] = []
            user_final[user_id]['spelling_errors'].append(err)

    # 5. Save Final with Statistics
    final_results = []
    # Only process users we actually have data for
    processed_user_ids = set(user_final.keys())
    
    for user_id in processed_user_ids:
        # Find original user data
        udata = users[user_id]
        reviews = udata.get('reviews', [])
        
        # Calculate total word count from all reviews
        total_words = 0
        for review in reviews:
            title = (review.get('title') or review.get('summary') or '').strip()
            text = (review.get('text') or review.get('reviewText') or '').strip()
            full_text = f"{title}. {text}" if title and text else (title or text)
            if full_text:
                total_words += len(full_text.split())
        
        # Get validated errors
        spelling_errors = user_final[user_id].get('spelling_errors', [])
        
        # Calculate statistics by dimension and category
        dimension_1_stats = {}  # Character & Word Variants
        dimension_2_stats = {}  # Grammar & Structural Logic
        
        for err in spelling_errors:
            cat_id = err.get('category_id', 0)
            cat_name = err.get('category_name', 'Unknown')
            dimension = err.get('dimension', 1)  # Default to dimension 1
            
            # Determine dimension based on category_id if not explicitly set
            if 'dimension' not in err:
                if cat_id in [1, 2, 3, 5, 6, 8, 9, 10]:
                    dimension = 1
                elif cat_id in [7, 11, 12, 13, 14, 15]:
                    dimension = 2
                else:
                    dimension = 1  # Default
            
            key = f"{cat_id}_{cat_name.lower().replace(' ', '_')}"
            
            if dimension == 1:
                dimension_1_stats[key] = dimension_1_stats.get(key, 0) + 1
            else:
                dimension_2_stats[key] = dimension_2_stats.get(key, 0) + 1
        
        # Calculate totals and error rate
        total_errors = len(spelling_errors)
        error_rate = (total_errors / total_words * 100) if total_words > 0 else 0.0
        
        # Build statistics object
        statistics = {
            "dimension_1_character_word_variants": {
                "total": sum(dimension_1_stats.values()),
                "by_category": dimension_1_stats
            },
            "dimension_2_grammar_structural": {
                "total": sum(dimension_2_stats.values()),
                "by_category": dimension_2_stats
            },
            "total_errors": total_errors,
            "total_words": total_words,
            "error_rate": round(error_rate, 2)
        }
        
        final_results.append({
            'user_id': user_id,
            'review_count': udata.get('review_count', len(reviews)),
            'total_words': total_words,
            'spelling_errors': spelling_errors,
            'statistics': statistics
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'results': final_results}, f, ensure_ascii=False, indent=2)
    print(f"✅ Batch results saved to {output_file}")
    
    # Print summary statistics
    if final_results:
        stats = final_results[0]['statistics']
        print(f"\n📊 Statistics Summary:")
        print(f"   Total Words: {stats['total_words']}")
        print(f"   Total Errors: {stats['total_errors']}")
        print(f"   Error Rate: {stats['error_rate']}% (errors per 100 words)")
        print(f"   Dimension 1 (Character & Word): {stats['dimension_1_character_word_variants']['total']} errors")
        print(f"   Dimension 2 (Grammar & Structural): {stats['dimension_2_grammar_structural']['total']} errors")


def test_specific_sentences_with_llm(llm, sampling_params):
    """Test specific sentences that contain spelling errors."""
    test_sentences = [
        "It actually is an entire skin care regimine containing a cleanser, a toner, a facial cream, an eye cream and overall moisurizers.",
        "Typically, my skin care routine is the same morning and night (wash, apply hyalauronic serum, and some kind of moistutrizer)."
    ]

    print("\n" + "="*80)
    print("🧪 TESTING SPECIFIC SENTENCES WITH SPELLING ERRORS")
    print("="*80)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Sentence {i}: {sentence}")

        # Step 1: Tokenize the sentence (same logic as in the main script)
        words = re.findall(r'\b\w+(?:\'\w+)*\b', sentence.lower())
        print(f"🔤 Tokenized words: {words}")
        print(f"🔢 Total words: {len(words)}")

        # Step 2: Create LLM prompt for spelling analysis
        candidate_str = ", ".join([f'"{word}"' for word in words])

        json_template = """
{
  "error_details": [
    {
      "word": "example_word",
      "error_type": "misspelled_word",
      "explanation": "brief explanation of the spelling mistake",
      "likely_correct": "correct_spelling",
      "context": "full sentence where the error appears",
      "sentence_idx": 0
    }
  ]
}"""

        system_msg = f"""You are an intelligent spelling error detector. Analyze ALL words in this candidate list and identify genuine spelling errors: {candidate_str}

INSTRUCTIONS:
- Examine each word in the candidate list above
- Consider the full sentence context: "{sentence.strip()}"
- Determine which words are REAL spelling errors that need correction

CLASSIFICATION RULES:
- ✅ CORRECT (do not report): Common words (the, a, and, but, is, are, was, were, have, has, had, do, does, did, will, would, can, could, should, may, might, must)
- ✅ CORRECT (do not report): Pronouns (I, you, he, she, it, we, they, me, my, your, his, her, our, their, this, that, these, those)
- ✅ CORRECT (do not report): Prepositions (in, on, at, to, for, of, with, by, from, into, onto, upon, about, above, below, between, among)
- ✅ CORRECT (do not report): Valid contractions (wasn't, doesn't, didn't, isn't, aren't, hasn't, haven't, won't, can't, shouldn't, couldn't, wouldn't)
- ✅ CORRECT (do not report): Recognized brand names that are intentionally misspelled (like "iPhone", "eBay") - but check for obvious typos
- ❌ ERROR: Words that are clearly misspelled, have wrong letters, missing/extra letters, or transpositions
- ❌ ERROR: Words that don't make sense in the sentence context

REPORT ONLY words that are genuinely misspelled and need correction. Be extremely critical - many words in this list may contain subtle typos or letter errors, even if they appear to be technical terms.

Respond with a JSON object containing ONLY validated spelling error details:{json_template}

STRICT REQUIREMENTS:
- Only include words that ACTUALLY APPEAR in the sentence above
- Only report words from the candidate list that are genuinely misspelled
- If a candidate word does not appear in the sentence, DO NOT include it
- If a candidate word appears in the sentence but is correctly spelled (including all contractions), DO NOT include it
- Return an empty array if no valid spelling errors are found

VALIDATION GUIDELINES:
- MANDATORY: Perform EXACT character-by-character comparison with standard spelling
- IGNORE ALL dictionary knowledge - ONLY compare letters
- RULE: If ANY letters differ from standard spelling, it IS an error
- UNIFIED STANDARD: ADDED, REMOVED, REPLACED, or REORDERED letters ALL = spelling errors
- MISSING LETTERS: Always errors (e.g., "definately" -> "definitely", "seperate" -> "separate")
- EXTRA LETTERS: Always errors (e.g., "hyalauronic" -> "hyaluronic", "moistutrizer" -> "moisturizer")
- LETTER SUBSTITUTIONS: Always errors (e.g., "recieve" -> "receive", "teh" -> "the")
- LETTER TRANSPOSITIONS: Always errors (e.g., "wierd" -> "weird")
- CRITICAL: "regimine" vs "regimen" = ERROR (missing 'e')
- CRITICAL: "moisurizers" vs "moisturizers" = ERROR (missing 't')
- Ignore punctuation-only differences
- STRICTLY check spelling of ALL words, including technical terms
- Do NOT give special treatment to technical or scientific terms - they must be spelled correctly too
- Common misspellings like "recieve"->"receive", "seperate"->"separate" should ALWAYS be flagged as errors
- FOR EACH CANDIDATE WORD: Explicitly verify if it appears in a standard English dictionary
- If a word is NOT in standard dictionary, it should be considered a spelling error
- Technical terms must also follow standard spelling conventions
- IGNORE CONTEXT INFLUENCE: Do NOT consider domain expertise or field specialization when evaluating spelling. A misspelled word is an error regardless of whether it appears in "skin care", "medical", or any other specialized context
- STRICT LETTER MATCHING: Perform mandatory character-by-character comparison against standard English dictionary spelling. ANY difference in letters (missing/extra/swapped/transposed) constitutes a spelling error, regardless of context
- DO NOT TRUST YOUR OWN DICTIONARY: Even if a word appears in your training data or internal knowledge base, perform explicit letter-by-letter verification against standard spelling
- MANDATORY LETTER-LEVEL VERIFICATION: For problematic words like "regimine", "moisurizers", "hyalauronic", "moistutrizer" - compare each character position against correct forms: "regimen", "moisturizers", "hyaluronic", "moisturizer". ANY mismatch constitutes an error, regardless of dictionary inclusion
- SPECIFIC CHECK: "regimine" (7 chars) vs "regimen" (7 chars) - position 6: 'e' vs 'n' = MISMATCH = ERROR
- SPECIFIC CHECK: "moisurizers" (11 chars) vs "moisturizers" (12 chars) - missing 't' = MISMATCH = ERROR

CHARACTER-LEVEL SPELLING AUDIT (MANDATORY):
For each candidate word, perform this exact process:
1. Identify the likely intended word (what it should be spelled as)
2. Compare character-by-character against the standard dictionary spelling
3. Flag ANY difference in letters (missing/extra/swapped/transposed)
4. Do NOT assume technical terms are exempt - check their spelling rigorously
5. Even if a word "sounds right" or is a valid brand, if letters differ from standard spelling, it is an ERROR

IMPORTANT: ALL LETTER DIFFERENCES ARE SPELLING ERRORS. Do NOT make exceptions based on whether letters are missing, extra, substituted, or transposed. Do NOT identify:
- Grammar errors (including articles a/an/the)
- Punctuation issues (exclamation marks, question marks, etc.)
- Sentence structure problems
- Style preferences
- Regional spelling variations (if they are valid alternatives)
- ANY issues where the alphabetic letters are correct but punctuation differs

CRITICAL RULE: If the only difference between the original word and your suggested correction is punctuation (like "!" or "?" or "."), DO NOT report it as a spelling error. This is a punctuation/style choice, not a spelling mistake.

FORBIDDEN PHRASES: Never use phrases like "error does not involve letter changes" or "the error does not involve actual letter changes" - ANY letter difference (missing/extra/swapped/transposed) IS a spelling error and MUST be reported.

  EXAMPLES of what TO flag:
  - "definately" -> "definitely" (missing 'i' - letter change)
  - "recieve" -> "receive" (wrong letters - e/i swapped)
  - "wierd" -> "weird" (wrong letter order - e/i swapped)
  - "seperate" -> "separate" (missing 'a' - letter missing)
  - "teh" -> "the" (wrong letters - e/h swapped)

  EXAMPLES of what NOT TO flag (punctuation/style issues):
  - "bar none!" -> "bar none" (only punctuation differs - NOT a spelling error)
  - "Good!" -> "Good" (only punctuation differs - NOT a spelling error)
  - "Hello?" -> "Hello" (only punctuation differs - NOT a spelling error)
  - "Yes." -> "Yes" (only punctuation differs - NOT a spelling error)

  REMEMBER: Spelling errors involve CHANGES TO LETTERS. Punctuation is not part of spelling."""

        prompt = f"""<s> [INST] {system_msg}

Sentence to analyze (#{i}):
```
{sentence}
```

VALIDATE these specific candidate words for spelling errors: {candidate_str}

For each candidate word, determine:
- Is this word actually misspelled (not in standard English dictionary)?
- Does the error involve actual letter changes (not just punctuation)?
- What type of spelling error is it?
- Perform character-by-character comparison with the correct spelling

MANDATORY SPELLING CHECK:
1. Break down each word into individual letters
2. Compare against the standard dictionary spelling
3. Flag any missing, extra, swapped, or transposed letters
4. Technical terms are NOT exempt - check them rigorously

IMPORTANT: Only confirm errors for words in the candidate list above. Do not find new errors.

Provide your validation results as a JSON object with "error_details" as specified.
[/INST]"""

        try:
            llm_response, ok = call_llm_with_retry(llm, prompt, context="test_sentence")
            if not ok:
                llm_response = "{}"

            print("\n🤖 LLM Spelling Analysis:")
            print(f"   {llm_response}")

            # Try to parse JSON response and provide detailed explanation for each error
            try:
                import json
                # Extract the LAST complete JSON object from LLM response (LLM may provide multiple JSON blocks)
                error_details = []

                # Find all potential JSON blocks (from last to first)
                last_json_start = -1
                search_pos = 0
                while True:
                    json_start = llm_response.find('{', search_pos)
                    if json_start == -1:
                        break
                    last_json_start = json_start
                    search_pos = json_start + 1

                if last_json_start != -1:
                    # Try to extract the last JSON block
                    json_content = llm_response[last_json_start:]
                    brace_count = 0
                    json_end = last_json_start

                    for i, char in enumerate(json_content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = last_json_start + i + 1
                                break

                    if json_end > last_json_start:
                        json_str = llm_response[last_json_start:json_end]
                        try:
                            response_json = json.loads(json_str)
                            error_details = response_json.get('error_details', [])
                        except json.JSONDecodeError:
                            # If the last JSON fails, try to find any valid JSON by going backwards
                            pass

                if error_details:
                    print(f"\n📖 Detailed Analysis of {len(error_details)} spelling errors:")
                    for error in error_details:
                        word = error.get('word', 'unknown')
                        correction = error.get('likely_correct', 'unknown')
                        explanation = error.get('explanation', 'no explanation')

                        # Generate detailed explanation for this specific error
                        detail_prompt = f"""Analyze this spelling error in detail:

Incorrect word: "{word}"
Correct word: "{correction}"
Context: "{sentence.strip()}"

Please provide a detailed explanation of:
1. Why "{word}" is incorrect
2. Why "{correction}" is the correct spelling
3. The linguistic or etymological reason for the correct form
4. Any common patterns or rules this illustrates

Be specific and educational."""

                        try:
                            detailed_explanation, ok = call_llm_with_retry(llm, detail_prompt, context="test_sentence_detail")
                            if not ok:
                                detailed_explanation = "Failed to generate detail."

                            print(f"\n   🔍 Error: '{word}' → '{correction}'")
                            print("   📖 Explanation:")
                            for line in detailed_explanation.split('\n'):
                                if line.strip():
                                    print(f"      {line}")
                        except Exception as detail_e:
                            print(f"   ⚠️ Failed to generate detailed explanation: {detail_e}")

                else:
                    print("   ✅ No spelling errors found in this sentence.")
            except json.JSONDecodeError as json_e:
                print(f"   ⚠️ Failed to parse LLM response as JSON: {json_e}")

        except Exception as e:
            print(f"❌ Error analyzing sentence {i}: {e}")

        print("-" * 80)

    print("\n" + "="*80)
    print("✅ SPECIFIC SENTENCES TESTING COMPLETED")
    print("="*80)


def main(args):
    """Main function."""
    # Detect PyTorch/CUDA availability and set environment for vllm accordingly
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
    except Exception:
        torch_available = False
        cuda_available = False

    if cuda_available:
        # Respect requested CUDA device when CUDA is available
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        print(f"Using CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        # No CUDA available — force CPU mode for vllm to avoid empty device string errors
        print("⚠️ No CUDA detected or torch not available. Forcing CPU mode for vllm.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["VLLM_DEVICE"] = "cpu"
    
    # Initialize model using model.py
    print(f"\n🚀 Initializing siliconflow model")
    llm = get_gm_model()
    print("✅ Model initialized")
    
    # Sampling parameters are now handled inside call_llm_with_retry or per model call
    sampling_params = None

    # Process users in batch mode
    asyncio.run(process_users_batch(
        args.input_file,
        args.output_file,
        max_users=args.max_users,
        start_idx=args.start_idx,
        min_reviews=args.min_reviews
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Review Style Analysis using Mistral-7B")
    parser.add_argument("--input-file", type=str, required=False, default=HARD_INPUT_FILE,
                       help="Input JSON file with user reviews (default: HARD_INPUT_FILE)")
    parser.add_argument("--output-file", type=str, required=False, default=HARD_OUTPUT_FILE,
                       help="Output JSON file for style analysis results (default: HARD_OUTPUT_FILE)")
    parser.add_argument("--hf-token", type=str, default=HF_TOKEN,
                       help="HuggingFace token")
    parser.add_argument("--cuda-device", type=int, default=0,
                       help="CUDA device ID")
    parser.add_argument("--max-users", type=int, default=5,
                       help="Maximum number of users to process (default: 5)")
    parser.add_argument("--min-reviews", type=int, default=5,
                       help="Minimum number of reviews a user must have to be processed (default: 5)")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Starting index for processing users")
    
    args = parser.parse_args()
    # Enforce hard-coded configuration: ignore any CLI or sbatch-provided inputs
    args.input_file = HARD_INPUT_FILE
    args.output_file = HARD_OUTPUT_FILE
    args.cuda_device = HARD_CUDA_DEVICE
    args.max_users = HARD_MAX_USERS
    args.min_reviews = HARD_MIN_REVIEWS
    args.start_idx = HARD_START_IDX
    print(f"USING HARDCODED CONFIG -> input_file: {args.input_file}, output_file: {args.output_file}, "
          f"cuda_device: {args.cuda_device}, max_users: {args.max_users}, min_reviews: {args.min_reviews}")

    main(args)

