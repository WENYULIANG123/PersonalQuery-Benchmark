#!/usr/bin/env python3
"""
Amazon Review Adversarial User Fingerprint Extraction using SiliconFlow LLM

This script extracts adversarial grammar fingerprints from Amazon beauty product reviews by:
1. Analyzing user writing habits across 4 adversarial dimensions
2. Identifying specific vocabulary and patterns that can make queries difficult
3. Generating user-specific adversarial fingerprints for robust testing

Adversarial Dimensions:
D1 - Syntactic Dilution (噪音与稀释): Fluff insertion, passive bureaucracy
D2 - Structural Complexity (结构复杂度): Long dependencies, negation traps
D3 - Referential Ambiguity (指代模糊): Generic nouns, pronoun-heavy text
D4 - Information Scattering (信息分散): Topic delay, circumlocution
"""

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Ensure stark/code is on Python path so we can import model.py
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
from model import get_gm_model

import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK punkt tokenizer if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Import spaCy for word tokenization
try:
    import spacy
except ImportError:
    print("Installing spacy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# ============================================================================
# Adversarial Grammar Pattern Definitions (For Performance Degradation)
# ============================================================================

GRAMMAR_DIMENSIONS = {
    'D1': {
        'name': 'Syntactic Dilution (噪音与稀释)',
        'patterns': {
            'FLUFF_INSERT': {
                'description': 'Excessive use of politeness, hesitation markers, or functional filler words that dilute keyword density',
                'features': ['hedging', 'excessive_politeness', 'filler_phrases'],
                'examples': ['I was wondering if it would be possible to...', 'Could you maybe tell me...']
            },
            'PASSIVE_BUREAU': {
                'description': 'Use of passive voice or nominalization that hides the agent or action, increasing cognitive load',
                'features': ['passive_voice', 'nominalization', 'formal_stiffness'],
                'examples': ['An application was made...', 'The item is desired by me']
            }
        }
    },
    'D2': {
        'name': 'Structural Complexity (结构复杂度)',
        'patterns': {
            'LONG_DEPENDENCY': {
                'description': 'Separating head nouns from their modifiers with long intervening clauses',
                'features': ['intervening_clauses', 'distant_modification'],
                'examples': ['The jacket that I saw yesterday which was red', 'The battery for the device that acts as a phone']
            },
            'NEGATION_TRAP': {
                'description': 'Use of double negatives, exclusion logic, or "not un-" constructions',
                'features': ['double_negation', 'litotes', 'exclusionary_phrasing'],
                'examples': ['It is not unlike...', 'I am not looking for non-waterproof...']
            }
        }
    },
    'D3': {
        'name': 'Referential Ambiguity (指代模糊)',
        'patterns': {
            'DE_ENTITY': {
                'description': 'Using generic terms (stuff, thing, item, gear) instead of specific entity names',
                'features': ['generic_nouns', 'loss_of_specificity'],
                'examples': ['I need stuff for my thing', 'looking for gear']
            },
            'PRONOUN_HEAVY': {
                'description': 'Heavy reliance on pronouns (it, they, one) without immediate local antecedents',
                'features': ['unclear_antecedents', 'pronoun_dominance'],
                'examples': ['I want the one that goes with it', 'Does it fit that?']
            }
        }
    },
    'D4': {
        'name': 'Information Scattering (信息分散)',
        'patterns': {
            'TOPIC_DELAY': {
                'description': 'Burying the core intent or main object at the very end of a long sentence',
                'features': ['delayed_subject', 'end_weight'],
                'examples': ['After considering all options and looking around, I want a hat.']
            },
            'CIRCUMLOCUTION': {
                'description': 'Talking around a concept using definitions instead of the word itself',
                'features': ['wordy_description', 'avoiding_direct_terms'],
                'examples': ['A device used for typing' '(instead of Keyboard)']
            }
        }
    }
}


# Simple configuration
INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/review_categories/All_Beauty.jsonl"
OUTPUT_FILE = "/home/wlia0047/ar57_scratch/wenyu/amazon_review_grammer_analysis.json"


# ============================================================================
# Helper Functions
# ============================================================================

def get_dimension_name(dimension_code):
    """Get dimension name from code."""
    dimension_names = {
        'D1': 'Structure Completeness',
        'D2': 'Logical Connection',
        'D3': 'Modification Preference',
        'D4': 'Feature Bias'
    }
    return dimension_names.get(dimension_code, 'Unknown')


def get_pattern_description(dimension_code, pattern_code):
    """Get pattern description from dimension and pattern codes."""
    if dimension_code in GRAMMAR_DIMENSIONS and pattern_code in GRAMMAR_DIMENSIONS[dimension_code]['patterns']:
        return GRAMMAR_DIMENSIONS[dimension_code]['patterns'][pattern_code]['description']
    return f'{pattern_code} pattern in {dimension_code}'


# ============================================================================
# COMMENTED OUT: Grammar Pattern Analysis Functions (No longer used)
# ============================================================================

# The following functions have been commented out as the script now focuses
# solely on adversarial user fingerprint extraction. These functions were used
# for the previous grammar pattern analysis but are no longer needed.

def analyze_grammar_patterns(text, llm_model):
    """Analyze text for grammar patterns across all 4 dimensions using LLM directly with concurrency."""
    patterns_found = []

    # Tokenize into sentences for processing
    print(f"🔄 Tokenizing text into sentences...", flush=True)
    sys.stdout.flush()
    sentences = sent_tokenize(text)
    print(f"[OK] Found {len(sentences)} sentences", flush=True)
    sys.stdout.flush()

    # Use concurrency: process each sentence independently with LLM
    max_concurrent_sentences = min(250, len(sentences))
    print(f"🔍 Starting concurrent LLM analysis of {len(sentences)} sentences (max {max_concurrent_sentences} concurrent)...", flush=True)
    sys.stdout.flush()

    def analyze_single_sentence_with_llm(sentence_data):
        """Analyze a single sentence with LLM."""
        sent_idx, sentence = sentence_data
        return analyze_single_sentence_llm(sentence, sent_idx, llm_model)

    # Prepare sentence data for concurrent processing
    sentence_data = [(i, sentence.strip()) for i, sentence in enumerate(sentences) if sentence.strip()]

    # Process with concurrency
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_concurrent_sentences) as executor:
        future_to_sentence = {
            executor.submit(analyze_single_sentence_with_llm, data): data[0]
            for data in sentence_data
        }

        processed_count = 0
        for future in as_completed(future_to_sentence):
            sent_idx = future_to_sentence[future]
            try:
                sentence_patterns = future.result()
                patterns_found.extend(sentence_patterns)
                processed_count += 1

                # Progress update
                if processed_count % 50 == 0 or processed_count == len(sentences):
                    print(f"📊 LLM analysis progress: {processed_count}/{len(sentences)} sentences processed, {len(patterns_found)} patterns found", flush=True)
                    sys.stdout.flush()

            except Exception as e:
                print(f"⚠️ Failed to analyze sentence {sent_idx}: {str(e)[:50]}...", flush=True)

    print(f"[OK] Grammar pattern analysis completed: found {len(patterns_found)} patterns across {len(sentences)} sentences", flush=True)
    sys.stdout.flush()
    return patterns_found


def analyze_single_sentence_llm(sentence, sent_idx, llm_model):
    """Analyze a single sentence for grammar patterns using LLM directly."""
    import time as time_module

    # Create ADVERSARIAL grammar analysis prompt
    grammar_dimensions_text = """
## Adversarial Grammar Dimensions (Features that degrade retrieval):

### D1 - Syntactic Dilution (噪音与稀释)
- FLUFF_INSERT: Excessive politeness/hedging/filler (e.g., "I was wondering if maybe...")
- PASSIVE_BUREAU: Passive voice or nominalization (e.g., "Selection was made by...")

### D2 - Structural Complexity (结构复杂度)
- LONG_DEPENDENCY: Modifiers separated far from nouns (e.g., "The box [clause] [clause] that is red")
- NEGATION_TRAP: Double negatives or exclusion logic (e.g., "Not unlike", "Anything but...")

### D3 - Referential Ambiguity (指代模糊)
- DE_ENTITY: Using generic nouns (stuff, thing, gear) instead of specific names
- PRONOUN_HEAVY: Overuse of pronouns (it, that, one) lacking clear context

### D4 - Information Scattering (信息分散)
- TOPIC_DELAY: Main intent buried at the very end of long sentence
- CIRCUMLOCUTION: Describing a word definition instead of using the word itself
"""

    analysis_prompt = f"""<s> [INST] ## Task: Identify Adversarial Grammar Patterns

You are a search engine optimization expert analyzing text for "Hard-to-Retrieve" patterns.
Analyze the following sentence and identify grammar patterns that would make this sentence DIFFICULT for a search engine (BM25 or Dense Retriever) to understand.

{grammar_dimensions_text}

## Sentence to Analyze:
"{sentence}"

## Instructions:
1. Look for patterns that introduce **Noise, Ambiguity, or Complexity**.
2. Identify ALL patterns from the list above that apply.
3. Provide specific evidence (e.g., which words create the ambiguity).
4. If the sentence is clear and simple, return an empty list.

## Output Format (JSON):
{{
  "sentence_idx": {sent_idx},
  "sentence": "{sentence}",
  "patterns_found": [
    {{
      "dimension": "D1",
      "pattern": "FLUFF_INSERT",
      "confidence": 0.9,
      "evidence": "Sentence starts with 10 words of polite padding 'I was just wondering if you could help me find' before the keyword."
    }}
  ]
}}

Return only valid JSON. [/INST]"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": analysis_prompt}]
            response = llm_model.invoke(messages)
            response_str = response.content.strip()

            # Parse JSON response
            json_str = parse_llm_json_response(response_str)
            if json_str:
                try:
                    analysis_result = json.loads(json_str)
                    patterns = analysis_result.get('patterns_found', [])

                    # Convert to the expected format
                    patterns_found = []
                    for pattern in patterns:
                        patterns_found.append({
                            'dimension': pattern.get('dimension', 'Unknown'),
                            'dimension_name': get_dimension_name(pattern.get('dimension', 'Unknown')),
                            'pattern': pattern.get('pattern', 'Unknown'),
                            'pattern_description': get_pattern_description(pattern.get('dimension', 'Unknown'), pattern.get('pattern', 'Unknown')),
                            'sentence_idx': sent_idx,
                            'sentence': sentence,
                            'confidence': pattern.get('confidence', 0.5),
                            'evidence': pattern.get('evidence', 'LLM detected pattern'),
                            'llm_validation': {
                                'validated': True,
                                'confidence': 'high',
                                'reason': 'Direct LLM detection'
                            }
                        })

                    return patterns_found

                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        time_module.sleep(1)
                        continue
                    else:
                        # Return empty list on failure
                        return []

        except Exception as e:
            if attempt < max_retries - 1:
                time_module.sleep(1)
                continue
            else:
                return []

    return []


def analyze_structure_completeness_spacy(tokens, doc):
    """Analyze D1 - Structure Completeness patterns using spaCy dependency parsing."""
    patterns = []

    # Use spaCy dependency parsing for accurate structure analysis
    has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in tokens)
    has_root_verb = any(token.pos_ == 'VERB' and token.dep_ == 'ROOT' for token in tokens)
    has_noun = any(token.pos_.startswith('N') for token in tokens)
    has_pronoun = any(token.pos_ == 'PRON' for token in tokens)

    # Get tense information from morphology
    verb_tenses = [token.morph.get('Tense') for token in tokens if token.pos_ == 'VERB']
    has_tense = any(tense and tense[0] in ['Past', 'Pres', 'Fut'] for tense in verb_tenses)

    # FULL_SENT: Complete sentence with proper subject-verb structure
    if has_subject and has_root_verb and has_noun:
        confidence = 0.9 if has_tense else 0.7
        tense_info = f" with tense indicators" if has_tense else " (tense unclear)"
        patterns.append({
            'pattern': 'FULL_SENT',
            'description': 'Complete sentence with subject-verb structure and clear syntactic roles',
            'confidence': confidence,
            'evidence': f"Found subject ({has_subject}), root verb ({has_root_verb}), nouns ({has_noun}){tense_info}"
        })

    # COMP_DROP: Missing explicit subject (imperative or subject-dropped constructions)
    # Check for verb-first patterns without clear subject
    if has_root_verb and not has_subject and not has_pronoun:
        # Look for imperative patterns (base form verbs at start)
        first_verb = next((token for token in tokens if token.pos_ == 'VERB'), None)
        if first_verb and first_verb.tag_ == 'VB':  # Base form
            patterns.append({
                'pattern': 'COMP_DROP',
                'description': 'Imperative or subject-dropped construction',
                'confidence': 0.8,
                'evidence': f"Root verb '{first_verb.text}' in base form without explicit subject"
            })

    # NOUN_STACK: Fragment consisting only of nouns and adjectives (no verb structure)
    if not has_root_verb and has_noun:
        noun_count = sum(1 for token in tokens if token.pos_.startswith('N'))
        adj_count = sum(1 for token in tokens if token.pos_ == 'ADJ')

        # Must be mostly nouns/adjectives with no verb structure
        if noun_count + adj_count >= len(tokens) * 0.8 and len(tokens) >= 2:
            patterns.append({
                'pattern': 'NOUN_STACK',
                'description': 'Fragment consisting only of nouns and adjectives without verb structure',
                'confidence': 0.9,
                'evidence': f"No verb structure, {noun_count} nouns, {adj_count} adjectives in {len(tokens)} tokens"
            })

    return patterns


def analyze_logical_connection_spacy(tokens, doc, token_text_set):
    """Analyze D2 - Logical Connection patterns using improved spaCy analysis."""
    patterns = []

    # ZERO_LINK: True fragmented structures (not complete sentences)
    # Only detect if there are multiple independent clauses without conjunctions
    sentences = [sent for sent in doc.sents]
    if len(sentences) > 1:
        # Check if sentences are truly fragments (not complete sentences)
        fragment_count = 0
        total_conjunctions = 0

        for sent in sentences:
            sent_tokens = [token for token in sent if not token.is_punct and not token.is_space]
            if len(sent_tokens) < 3:  # Very short fragments
                fragment_count += 1
                continue

            # Check if this sentence has a subject and verb (complete sentence)
            has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in sent_tokens)
            has_verb = any(token.pos_ == 'VERB' for token in sent_tokens)

            if not (has_subject and has_verb):
                fragment_count += 1

            # Count conjunctions between sentences
            sent_text = sent.text.strip()
            if sent_text.endswith(('.', '!', '?')):
                # Look for conjunctions at the beginning of this sentence
                first_word = sent_tokens[0].text.lower() if sent_tokens else ""
                if first_word in ['and', 'but', 'or', 'so', 'because', 'although']:
                    total_conjunctions += 1

        # Only classify as ZERO_LINK if we have true fragments without conjunctions
        if fragment_count >= 2 and total_conjunctions == 0 and len(sentences) >= 2:
            patterns.append({
                'pattern': 'ZERO_LINK',
                'description': 'Uses fragmented structure with multiple independent clauses without conjunctions',
                'confidence': 0.7,
                'evidence': f"Found {fragment_count} fragments in {len(sentences)} sentences, no connecting conjunctions"
            })

    # COORD_LINK: Coordinating conjunctions (and, but, or, etc.)
    # FIXED: Use strict token-level matching to prevent substring false positives
    coord_conjunctions = ['and', 'but', 'or', 'nor', 'yet', 'so']
    coord_tokens = []

    # Use strict token-level matching - only exact token matches
    for token in tokens:
        token_text = token.text.lower()
        if token_text in coord_conjunctions and token.pos_ == 'CCONJ':
            coord_tokens.append(token)

        if coord_tokens:
            found_coords = list(set(token.text.lower() for token in coord_tokens))
            patterns.append({
                'pattern': 'COORD_LINK',
                'description': 'Uses coordinating conjunctions to connect equal grammatical elements',
                'confidence': 0.9,
                'evidence': f"Found coordinating conjunctions: {found_coords} ({len(coord_tokens)} instances)"
            })

    # SUB_LINK: Subordinating conjunctions (because, if, when, etc.)
    # FIXED: Enhanced validation with strict token matching to prevent false positives
    sub_conjunctions = ['because', 'if', 'when', 'which', 'that', 'who', 'where', 'why', 'how', 'whether', 'while', 'although', 'though', 'since']
    sub_tokens = []

    for token in tokens:
        token_text = token.text.lower()
        if token_text in sub_conjunctions and token.pos_ == 'SCONJ':
            # Exclude 'that' when it's a complementizer (direct object of verb)
            if token_text == 'that':
                # Check if 'that' introduces a noun clause as direct object
                if token.dep_ == 'mark' and token.head.pos_ == 'VERB':
                    continue  # Skip complementizer 'that'
            sub_tokens.append(token)

        if sub_tokens:
            sub_words = list(set(token.text.lower() for token in sub_tokens))
            patterns.append({
                'pattern': 'SUB_LINK',
                'description': 'Uses subordinating conjunctions for complex sentence relationships',
                'confidence': 0.9,
                'evidence': f"Found subordinating conjunctions: {sub_words} ({len(sub_tokens)} instances)"
            })

    # PREP_LINK: Prepositions used for adverbial modification of verbs (improved detection)
    # FIXED: More restrictive criteria to avoid over-classification of PREP_LINK
    prep_adverbial = []

    for token in tokens:
        if token.pos_ == 'ADP':  # Preposition
            head = token.head

            # Exclude phrasal verb particles (prt dependency)
            if token.dep_ == 'prt' and head.pos_ == 'VERB':
                continue

            # Exclude prepositions that modify nouns (post-modification)
            if head.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'prep':
                continue

            # Only include true adverbial prepositions modifying verbs
            if head.pos_ == 'VERB' and token.dep_ in ['prep', 'advmod']:
                # FIXED: More restrictive - exclude very common prepositions that are rarely truly logical
                token_text = token.text.lower()
                if token_text not in ['to', 'of', 'at', 'in', 'on', 'by', 'for', 'with']:
                    # Additional semantic check: prefer prepositions that indicate logical relationships
                    logical_preps = ['after', 'before', 'during', 'since', 'until', 'while', 'without', 'despite', 'although']
                    if token_text in logical_preps or len(token_text) > 2:  # Longer preps more likely to be logical
                        prep_adverbial.append(token_text)

    if prep_adverbial:
        unique_preps = list(set(prep_adverbial))
        patterns.append({
            'pattern': 'PREP_LINK',
            'description': 'Uses prepositions for adverbial modification of verbs/actions',
            'confidence': 0.8,
            'evidence': f"Found adverbial prepositions: {unique_preps} ({len(prep_adverbial)} instances)"
        })

    return patterns


def analyze_modification_preference_spacy(tokens, doc):
    """Analyze D3 - Modification Preference patterns using improved spaCy dependency parsing."""
    patterns = []

    # PRE_MOD: Multiple pre-modifying adjectives before nouns (improved detection)
    noun_pre_modifiers = {}

    for token in tokens:
        if token.pos_ in ['NOUN', 'PROPN']:  # Include proper nouns
            # Find all adjectives that directly modify this noun
            direct_modifiers = []
            for child in token.children:
                if child.pos_ == 'ADJ' and child.dep_ == 'amod':
                    direct_modifiers.append(child)

            # Find consecutive adjectives immediately before the noun
            consecutive_modifiers = []
            current = token
            while current.i > 0:  # Check tokens before current
                prev_token = doc[current.i - 1]
                if prev_token.pos_ == 'ADJ':
                    consecutive_modifiers.insert(0, prev_token)
                    current = prev_token
                else:
                    break

            # Combine direct and consecutive modifiers, remove duplicates
            all_modifiers = list(set(direct_modifiers + consecutive_modifiers))

            # Only count as PRE_MOD if there are 2+ distinct descriptive adjectives
            # Exclude determiners and quantifiers that aren't descriptive
            descriptive_adjs = [adj for adj in all_modifiers
                              if adj.text.lower() not in ['a', 'an', 'the', 'this', 'that', 'these', 'those']
                              and not adj.text.isdigit()  # Exclude pure numbers
                              and len(adj.text) > 2]  # Exclude very short words

            if len(descriptive_adjs) >= 2:
                noun_pre_modifiers[token.text] = descriptive_adjs

    if noun_pre_modifiers:
        total_instances = len(noun_pre_modifiers)
        total_adjectives = sum(len(adjs) for adjs in noun_pre_modifiers.values())
        patterns.append({
            'pattern': 'PRE_MOD',
            'description': 'Stacks multiple descriptive adjectives before nouns for detailed description',
            'confidence': 0.85,
            'evidence': f"Found {total_instances} nouns with 2+ descriptive adjectives, total {total_adjectives} adjectives"
        })

    # POST_MOD: Postpositional modifiers (prepositional phrases after nouns)
    post_modifiers = []

    for token in tokens:
        if token.pos_ == 'ADP':  # Preposition
            head = token.head
            # Only count if preposition modifies a noun (not verb) and is clearly post-modifying
            if (head.pos_ in ['NOUN', 'PROPN'] and
                token.dep_ == 'prep' and
                token.text.lower() not in ['to', 'of']):  # Exclude common grammatical preps

                # Check if this creates a meaningful post-modification
                prep_phrase = token.text
                # Look for objects of the preposition
                for child in token.children:
                    if child.dep_ in ['pobj', 'dobj']:
                        prep_phrase += f" {child.text}"
                        break

                post_modifiers.append({
                    'preposition': token.text,
                    'head_noun': head.text,
                    'phrase': prep_phrase
                })

    if post_modifiers:
        unique_preps = list(set(mod['preposition'] for mod in post_modifiers))
        patterns.append({
            'pattern': 'POST_MOD',
            'description': 'Uses prepositional phrases as post-modifiers for nouns',
            'confidence': 0.8,
            'evidence': f"Found post-modifying prepositions: {unique_preps} ({len(post_modifiers)} instances)"
        })

    # INTENSIFIER: Degree intensifiers modifying adjectives or adverbs (improved detection)
    intensifier_words = ['very', 'too', 'extremely', 'so', 'really', 'quite', 'highly', 'super', 'ultra', 'totally', 'rather', 'pretty', 'fairly']

    valid_intensifiers = []
    for intensifier in intensifier_words:
        intensifier_tokens = [token for token in tokens if token.text.lower() == intensifier]

        for intensifier_token in intensifier_tokens:
            # Check if it modifies an adjective or adverb
            modifies_adj_adv = False
            for child in intensifier_token.children:
                if child.pos_ in ['ADJ', 'ADV'] and child.dep_ in ['advmod', 'amod']:
                    modifies_adj_adv = True
                    break

            # Also check if it's immediately followed by adj/adv
            if intensifier_token.i < len(doc) - 1:
                next_token = doc[intensifier_token.i + 1]
                if next_token.pos_ in ['ADJ', 'ADV']:
                    modifies_adj_adv = True

            if modifies_adj_adv:
                valid_intensifiers.append(intensifier)

    if valid_intensifiers:
        unique_intensifiers = list(set(valid_intensifiers))
        patterns.append({
            'pattern': 'INTENSIFIER',
            'description': 'Uses intensifiers to emphasize adjectives and adverbs',
            'confidence': 0.9,
            'evidence': f"Found intensifiers modifying adj/adv: {unique_intensifiers} ({len(valid_intensifiers)} instances)"
        })

    return patterns


def analyze_feature_bias_spacy(tokens, doc, token_text_set, original_sentence):
    """Analyze D4 - Feature Bias patterns using spaCy with token-level matching."""
    patterns = []

    # ART_LOSS: Missing articles before nouns
    if tokens:
        first_token = tokens[0]
        # Check if sentence starts with a common noun without article
        if (first_token.pos_.startswith('N') and
            first_token.text[0].isupper() and  # Capitalized (likely proper noun or sentence start)
            first_token.text.lower() not in ['a', 'an', 'the'] and
            not first_token.is_sent_start):  # Not at sentence boundary

            # Check if this could be a context where article is expected
            # Look for determiners in the vicinity
            has_nearby_det = any(child.pos_ == 'DET' for child in first_token.children)
            has_left_det = any(token.pos_ == 'DET' for token in first_token.lefts)

            if not has_nearby_det and not has_left_det:
                patterns.append({
                    'pattern': 'ART_LOSS',
                    'description': 'Omits articles before nouns where they might be expected',
                    'confidence': 0.6,
                    'evidence': f"Starts with noun '{first_token.text}' without preceding article"
                })

    # SHORTHAND: Abbreviations and shorthand notation
    shorthand_patterns = ['w/', 'w/o', 'pkg', 'info', 'qty', 'min', 'max', 'avg', 'btw', 'aka', 'etc', 'vs', 'esp', 'imo']
    found_shorthand = [abbr for abbr in shorthand_patterns if abbr in token_text_set]

    if found_shorthand:
        patterns.append({
            'pattern': 'SHORTHAND',
            'description': 'Uses abbreviations and shorthand notation',
            'confidence': 0.9,
            'evidence': f"Found shorthand abbreviations: {found_shorthand}"
        })

    # IMPERATIVE: Imperative mood (command form without subject)
    if tokens and len(tokens) > 0:
        first_token = tokens[0]
        # Check for base form verb at sentence start
        if (first_token.pos_ == 'VERB' and
            first_token.tag_ == 'VB' and  # Base form
            first_token.dep_ == 'ROOT' and  # Root of sentence
            not original_sentence.strip().endswith('?')):  # Not a question

            # Verify no explicit subject (imperative)
            has_explicit_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in tokens)
            if not has_explicit_subject:
                patterns.append({
                    'pattern': 'IMPERATIVE',
                    'description': 'Uses imperative mood with base verb form and no explicit subject',
                    'confidence': 0.8,
                    'evidence': f"Root verb '{first_token.text}' in base form without explicit subject"
                })

    return patterns


def validate_sentence_patterns(sentence_data, llm_model, start_time):
    """Validate all patterns for a single sentence in one LLM call."""
    import time as time_module

    sentence_key, sentence_patterns = sentence_data
    sent_idx, sentence = sentence_key.split(': ', 1)
    sent_idx = int(sent_idx)

    max_retries = 3
    timeout_per_call = 60  # Longer timeout for complex validation

    for attempt in range(max_retries):
        try:
            # Removed verbose logging - only show progress every 10 sentences

            # Build comprehensive validation prompt with all patterns for this sentence
            patterns_list = []
            for i, pattern in enumerate(sentence_patterns, 1):
                patterns_list.append(f"""
**Pattern {i}:**
- Dimension: {pattern['dimension']} ({pattern['dimension_name']})
- Pattern: {pattern['pattern']}
- Description: {pattern['pattern_description']}
- First Stage Confidence: {pattern['confidence']}
- First Stage Evidence: {pattern['evidence']}
""")

            patterns_text = "\n".join(patterns_list)

            validation_prompt = f"""<s> [INST] ## Task: Validate Adversarial Grammar Patterns

You are a linguistics and information retrieval expert. Review the "Adversarial/Hard-to-Retrieve" grammar patterns detected for this sentence.

## Sentence to Analyze:
"{sentence}"

## Detected Adversarial Patterns:
{patterns_text}

## Validation Task:
For EACH pattern listed above, determine if it correctly applies to the sentence. Consider:

### [OK] CONFIRM patterns that are correct:
- The pattern genuinely applies to the sentence
- The evidence accurately supports the pattern
- The classification fits linguistic standards

### [FAIL] REJECT patterns that have issues:
- Pattern does not actually apply to this sentence
- Evidence is incorrect or misleading
- Classification contradicts grammatical rules
- Pattern is a false positive from the first-stage analysis

### 🤔 CONSIDER:
- Informal/colloquial language usage
- Domain-specific terminology (beauty product reviews)
- Complex sentence structures
- Multiple valid interpretations

## Response Format (JSON):
{{
  "sentence_validation": {{
    "sentence_idx": {sent_idx},
    "sentence": "{sentence}",
    "pattern_validations": [
      {{
        "pattern_index": 1,
        "is_valid": true/false,
        "validation_confidence": "high/medium/low",
        "validation_reason": "Detailed explanation",
        "improved_evidence": "Optional: better evidence if original was inadequate"
      }},
      {{
        "pattern_index": 2,
        "is_valid": true/false,
        "validation_confidence": "high/medium/low",
        "validation_reason": "Detailed explanation"
      }}
      // ... for each pattern
    ]
  }}
}}

Provide thorough linguistic analysis for each pattern decision. [/INST]"""

            messages = [{"role": "user", "content": validation_prompt}]

            # Add timeout protection for LLM call
            start_call_time = time_module.time()
            result = [None]
            error = [None]

            def call_llm():
                try:
                    result[0] = llm_model.invoke(messages)
                except Exception as e:
                    error[0] = e

            import threading
            thread = threading.Thread(target=call_llm)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_per_call)

            if thread.is_alive():
                # Removed verbose timeout logging
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time_module.sleep(wait_time)
                    continue
                else:
                    # On timeout, reject all patterns for this sentence
                    for pattern in sentence_patterns:
                        pattern['second_stage_validation'] = {
                            'validated': False,
                            'validation_confidence': 'unknown',
                            'validation_reason': f'LLM timeout after {max_retries} attempts',
                            'consistency_check': False
                        }
                    return sentence_patterns

            if error[0]:
                raise error[0]

            response = result[0]
            response_str = response.content.strip()

            # Parse JSON response
            json_str = parse_llm_json_response(response_str)
            if json_str:
                try:
                    validation_result = json.loads(json_str)
                    sentence_validation = validation_result.get('sentence_validation', {})
                    pattern_validations = sentence_validation.get('pattern_validations', [])

                    # Apply validation results to each pattern
                    validated_sentence_patterns = []
                    for i, pattern in enumerate(sentence_patterns):
                        # Find corresponding validation result
                        validation = None
                        for pv in pattern_validations:
                            if pv.get('pattern_index') == (i + 1):
                                validation = pv
                                break

                        if validation:
                            is_valid = validation.get('is_valid', False)
                            validation_confidence = validation.get('validation_confidence', 'unknown')
                            validation_reason = validation.get('validation_reason', 'No reason provided')

                            pattern['second_stage_validation'] = {
                                'validated': is_valid,
                                'validation_confidence': validation_confidence,
                                'validation_reason': validation_reason,
                                'improved_evidence': validation.get('improved_evidence', ''),
                                'consistency_check': True
                            }

                            if is_valid:
                                validated_sentence_patterns.append(pattern)
                                # Removed verbose pattern confirmation logging
                            # Removed verbose pattern rejection logging
                        else:
                            # If no validation result found, assume invalid
                            pattern['second_stage_validation'] = {
                                'validated': False,
                                'validation_confidence': 'unknown',
                                'validation_reason': 'No validation result from LLM',
                                'consistency_check': False
                            }
                            print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ Pattern {i+1} missing validation result", flush=True)

                    return validated_sentence_patterns

                except json.JSONDecodeError as e:
                    # Removed verbose JSON parse error logging
                    if attempt < max_retries - 1:
                        time_module.sleep(2)
                        continue
                    else:
                        # On JSON error, reject all patterns
                        for pattern in sentence_patterns:
                            pattern['second_stage_validation'] = {
                                'validated': False,
                                'validation_confidence': 'unknown',
                                'validation_reason': f'JSON parsing failed: {str(e)}',
                                'consistency_check': False
                            }
                        return sentence_patterns

            # No JSON found
            # Removed verbose no JSON found logging
            if attempt < max_retries - 1:
                time_module.sleep(2)
                continue
            else:
                # On failure, reject all patterns
                for pattern in sentence_patterns:
                    pattern['second_stage_validation'] = {
                        'validated': False,
                        'validation_confidence': 'unknown',
                        'validation_reason': 'No valid JSON response from LLM',
                        'consistency_check': False
                    }
                return sentence_patterns

        except Exception as e:
            print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ Sentence validation failed (attempt {attempt + 1}): {str(e)[:50]}...", flush=True)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time_module.sleep(wait_time)
                continue
            else:
                # On failure, reject all patterns
                for pattern in sentence_patterns:
                    pattern['second_stage_validation'] = {
                        'validated': False,
                        'validation_confidence': 'unknown',
                        'validation_reason': f'Validation failed: {str(e)}',
                        'consistency_check': False
                    }
                return sentence_patterns

    return sentence_patterns


def validate_grammar_patterns_by_sentence(patterns, llm_model, start_time=0):
    """Validate grammar patterns by sentence: one LLM call per sentence with all detected patterns."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not patterns:
        return patterns

    # Group patterns by sentence for validation
    patterns_by_sentence = {}
    for pattern in patterns:
        sent_idx = pattern['sentence_idx']
        sentence = pattern['sentence']
        key = f"{sent_idx}: {sentence}"
        if key not in patterns_by_sentence:
            patterns_by_sentence[key] = []
        patterns_by_sentence[key].append(pattern)

    validated_patterns = []
    total_sentences = len(patterns_by_sentence)

    # Use concurrency for sentence-level validation (one LLM call per sentence)
    max_concurrent_sentences = min(100, total_sentences)

    print(f"[{time.time() - start_time:.1f}s] 🔍 Starting second-stage LLM validation: validating {total_sentences} sentences with all their patterns (max {max_concurrent_sentences} concurrent), one LLM call per sentence...", flush=True)
    sys.stdout.flush()

    # Process sentences concurrently
    sentence_data = list(patterns_by_sentence.items())
    with ThreadPoolExecutor(max_workers=max_concurrent_sentences) as executor:
        future_to_sentence = {}
        for sentence_idx, data in enumerate(sentence_data):
            future = executor.submit(validate_sentence_patterns, data, llm_model, start_time)
            future_to_sentence[future] = sentence_idx

        processed_sentences = 0
        validated_count = 0
        rejected_count = 0

        for future in as_completed(future_to_sentence):
            sentence_idx = future_to_sentence[future]
            processed_sentences += 1

            try:
                sentence_patterns = future.result()
                validated_count += len(sentence_patterns)
                validated_patterns.extend(sentence_patterns)

                # Progress update
                if processed_sentences % 10 == 0 or processed_sentences == total_sentences:
                    print(f"[{time.time() - start_time:.1f}s] 📊 Sentence validation progress: {processed_sentences}/{total_sentences} sentences processed, {validated_count} patterns confirmed", flush=True)
                    sys.stdout.flush()

            except Exception as e:
                # Removed verbose sentence validation failure logging
                # Conservative approach: include all patterns from failed sentence
                sent_key, sentence_patterns = sentence_data[sentence_idx]
                for pattern in sentence_patterns:
                    pattern['second_stage_validation'] = {
                        'validated': False,
                        'validation_confidence': 'unknown',
                        'validation_reason': f'Sentence validation failed: {str(e)}',
                        'consistency_check': False
                    }
                validated_patterns.extend(sentence_patterns)
                rejected_count += len(sentence_patterns)

    print(f"[{time.time() - start_time:.1f}s] [OK] Second-stage LLM validation complete: {len(validated_patterns)} patterns from {total_sentences} sentences", flush=True)
    print(f"[{time.time() - start_time:.1f}s] 📊 Final validation summary: {validated_count} patterns confirmed, {rejected_count} patterns rejected", flush=True)
    sys.stdout.flush()

    return validated_patterns

    def validate_single_pattern(pattern, pattern_idx, total_patterns, llm_model, start_time):
        """Validate a single grammar pattern from first stage analysis."""
        import time as time_module

        max_retries = 3
        timeout_per_call = 30

        for attempt in range(max_retries):
            try:
                print(f"[{time_module.time() - start_time:.1f}s] 🔍 Validating pattern {pattern_idx + 1}/{total_patterns}: {pattern['dimension']}-{pattern['pattern']}" + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""), flush=True)

                # Direct pattern validation prompt - validate the specific pattern identified in first stage
                validation_prompt = f"""<s> [INST] ## Task: Validate Specific Grammar Pattern Classification

You are a linguistics expert. Evaluate the following grammar pattern classification from a previous analysis.

## Pattern to Evaluate:
- **Dimension**: {pattern['dimension']} ({pattern['dimension_name']})
- **Pattern**: {pattern['pattern']}
- **Description**: {pattern['pattern_description']}
- **Sentence**: "{pattern['sentence']}"
- **First Stage Confidence**: {pattern['confidence']}
- **First Stage Evidence**: {pattern['evidence']}

## Validation Task:
Determine if this specific grammar pattern classification is accurate for the given sentence. Consider:

### [OK] CONFIRM if the pattern classification is correct:
- The pattern genuinely applies to the sentence
- The evidence accurately supports the pattern
- The classification fits linguistic standards
- No significant errors in the analysis

### [FAIL] REJECT if the pattern classification has issues:
- Pattern does not actually apply to this sentence
- Evidence is incorrect or misleading
- Classification contradicts grammatical rules
- Pattern represents a different grammatical phenomenon

### 🤔 CONSIDER EDGE CASES:
- Informal/colloquial language usage
- Domain-specific terminology
- Creative or non-standard grammatical constructions

## Response Format (JSON):
{{
  "is_valid": true/false,
  "validation_confidence": "high/medium/low",
  "validation_reason": "Detailed explanation of your validation decision",
  "improved_evidence": "Optional: better evidence if the original was inadequate",
  "suggested_corrections": "Optional: suggestions for improvement if pattern is invalid"
}}

Be thorough in your analysis and provide specific linguistic reasoning. [/INST]"""

                messages = [{"role": "user", "content": validation_prompt}]

                # Add timeout protection
                start_call_time = time_module.time()
                result = [None]
                error = [None]

                def call_llm():
                    try:
                        result[0] = llm_model.invoke(messages)
                    except Exception as e:
                        error[0] = e

                import threading
                thread = threading.Thread(target=call_llm)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout_per_call)

                if thread.is_alive():
                    print(f"[{time_module.time() - start_time:.1f}s]    ⏰ LLM validation timeout after {timeout_per_call}s (attempt {attempt + 1})", flush=True)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[{time_module.time() - start_time:.1f}s]    🔄 Retrying in {wait_time}s...", flush=True)
                        time_module.sleep(wait_time)
                        continue
                    else:
                        # Timeout failure
                        pattern['second_stage_validation'] = {
                            'validated': False,
                            'validation_confidence': 'unknown',
                            'validation_reason': f'LLM timeout after {max_retries} attempts',
                            'consistency_check': False
                        }
                        return pattern

                if error[0]:
                    raise error[0]

                response = result[0]
                response_str = response.content.strip()

                # Parse JSON response
                json_str = parse_llm_json_response(response_str)
                if json_str:
                    try:
                        validation_result = json.loads(json_str)
                        is_valid = validation_result.get('is_valid', False)
                        validation_confidence = validation_result.get('validation_confidence', 'unknown')
                        validation_reason = validation_result.get('validation_reason', 'No reason provided')

                        # Add validation results to pattern
                        pattern['second_stage_validation'] = {
                            'validated': is_valid,
                            'validation_confidence': validation_confidence,
                            'validation_reason': validation_reason,
                            'improved_evidence': validation_result.get('improved_evidence', ''),
                            'suggested_corrections': validation_result.get('suggested_corrections', ''),
                            'consistency_check': True
                        }

                        if is_valid:
                            print(f"[{time_module.time() - start_time:.1f}s]    [OK] Confirmed (confidence: {validation_confidence})", flush=True)
                        else:
                            print(f"[{time_module.time() - start_time:.1f}s]    [FAIL] Rejected (confidence: {validation_confidence}) - {validation_reason[:50]}...", flush=True)

                        return pattern

                    except json.JSONDecodeError as e:
                        print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ JSON parse error in validation (attempt {attempt + 1}): {str(e)}", flush=True)
                        if attempt < max_retries - 1:
                            time_module.sleep(1)
                            continue
                        else:
                            pattern['second_stage_validation'] = {
                                'validated': False,
                                'validation_confidence': 'unknown',
                                'validation_reason': f'JSON parsing failed: {str(e)}',
                                'consistency_check': False
                            }
                            return pattern

                # No JSON found
                print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ No JSON in validation response (attempt {attempt + 1})", flush=True)
                if attempt < max_retries - 1:
                    time_module.sleep(1)
                    continue
                else:
                    pattern['second_stage_validation'] = {
                        'validated': False,
                        'validation_confidence': 'unknown',
                        'validation_reason': 'No valid JSON response from LLM',
                        'consistency_check': False
                    }
                    return pattern

            except Exception as e:
                print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ Validation failed (attempt {attempt + 1}): {str(e)[:50]}...", flush=True)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time_module.sleep(wait_time)
                    continue
                else:
                    pattern['second_stage_validation'] = {
                        'validated': False,
                        'validation_confidence': 'unknown',
                        'validation_reason': f'Validation failed: {str(e)}',
                        'consistency_check': False
                    }
                    return pattern

        return pattern

    # Process patterns concurrently
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_pattern = {}
        for pattern_idx, pattern in enumerate(patterns):
            future = executor.submit(validate_single_pattern, pattern, pattern_idx, len(patterns), llm_model, start_time)
            future_to_pattern[future] = pattern_idx

        processed_validations = 0
        validated_count = 0

        for future in as_completed(future_to_pattern):
            pattern_idx = future_to_pattern[future]
            processed_validations += 1

            try:
                validated_pattern = future.result()
                validated_patterns.append(validated_pattern)

                if validated_pattern.get('second_stage_validation', {}).get('validated', False):
                    validated_count += 1

                # Progress update
                if processed_validations % 50 == 0 or processed_validations == len(patterns):
                    print(f"[{time.time() - start_time:.1f}s] 📊 Second-stage validation progress: {processed_validations}/{len(patterns)} processed, {validated_count} confirmed", flush=True)
                    sys.stdout.flush()

            except Exception as e:
                print(f"⚠️ Failed validation for pattern {pattern_idx + 1}: {str(e)[:50]}...", flush=True)
                # Add original pattern with error note
                pattern_copy = patterns[pattern_idx].copy()
                pattern_copy['second_stage_validation'] = {
                    'validated': False,
                    'validation_confidence': 'unknown',
                    'validation_reason': str(e),
                    'consistency_check': False
                }
                validated_patterns.append(pattern_copy)

    print(f"[{time.time() - start_time:.1f}s] [OK] Second-stage LLM validation complete: {len(validated_patterns)} patterns individually validated", flush=True)
    sys.stdout.flush()

    return validated_patterns

    def validate_single_pattern(pattern, pattern_idx, total_patterns, llm_model, start_time):
        """Validate a single grammar pattern with LLM (with timeout and retry)."""
        import time as time_module

        max_retries = 3
        timeout_per_call = 30  # 30 seconds timeout per LLM call

        for attempt in range(max_retries):
            try:
                print(f"[{time_module.time() - start_time:.1f}s] 🔍 Validating pattern {pattern_idx + 1}/{total_patterns}: {pattern['dimension']}-{pattern['pattern']}" + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""), flush=True)

                validation_prompt = f'''<s> [INST] ## Task: Validate Grammar Pattern Classification

**Pattern to Validate:**
- Dimension: {pattern['dimension']} ({pattern['dimension_name']})
- Pattern: {pattern['pattern']}
- Description: {pattern['pattern_description']}
- Sentence: "{pattern['sentence']}"
- Evidence: {pattern['evidence']}
- Confidence: {pattern['confidence']}

**Validation Task:**
Determine if this grammar pattern classification is accurate. Consider:

[CONFIRM] if the pattern correctly matches the grammatical structure
[REJECT] if the classification doesn't fit the actual grammar

**Response Format:**
{{
  "is_valid": true/false,
  "confidence": "high/medium/low",
  "reason": "brief explanation of validation decision"
}}
[/INST]'''

                messages = [{"role": "user", "content": validation_prompt}]

                # Add timeout protection for LLM call
                start_call_time = time_module.time()
                try:
                    # Note: Most LLM libraries don't support timeout parameter in invoke()
                    # We'll implement a manual timeout using threading
                    import threading

                    result = [None]
                    error = [None]

                    def call_llm():
                        try:
                            result[0] = llm_model.invoke(messages)
                        except Exception as e:
                            error[0] = e

                    thread = threading.Thread(target=call_llm)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=timeout_per_call)

                    if thread.is_alive():
                        print(f"[{time_module.time() - start_time:.1f}s]    ⏰ LLM call timeout after {timeout_per_call}s (attempt {attempt + 1})", flush=True)
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            print(f"[{time_module.time() - start_time:.1f}s]    🔄 Retrying in {wait_time}s...", flush=True)
                            time_module.sleep(wait_time)
                            continue
                        else:
                            # On timeout, reject all patterns for this sentence
                            for pattern in sentence_patterns:
                                pattern['second_stage_validation'] = {
                                    'validated': False,
                                    'validation_confidence': 'unknown',
                                    'validation_reason': f'LLM timeout after {max_retries} attempts',
                                    'consistency_check': False
                                }
                            return sentence_patterns

                except Exception as e:
                    # Removed verbose sentence validation failure logging
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time_module.sleep(wait_time)
                        continue
                    else:
                        # On failure, reject all patterns
                        for pattern in sentence_patterns:
                            pattern['second_stage_validation'] = {
                                'validated': False,
                                'validation_confidence': 'unknown',
                                'validation_reason': f'Validation failed: {str(e)}',
                                'consistency_check': False
                            }
                        return sentence_patterns

                if error[0]:
                    raise error[0]

                response = result[0]
                response_str = response.content.strip()

                # Parse JSON response
                json_str = parse_llm_json_response(response_str)
                if json_str:
                    try:
                        validation_result = json.loads(json_str)
                        is_valid = validation_result.get('is_valid', True)
                        confidence = validation_result.get('confidence', 'medium')
                        reason = validation_result.get('reason', 'LLM validation completed')

                        if is_valid:
                            pattern['llm_validation'] = {
                                'validated': True,
                                'confidence': confidence,
                                'reason': reason
                            }
                            return pattern, True
                        else:
                            print(f"[{time_module.time() - start_time:.1f}s]    [FAIL] LLM rejected pattern {pattern['dimension']}-{pattern['pattern']}: {reason[:100]}...", flush=True)
                            return pattern, False

                    except json.JSONDecodeError:
                        print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ JSON parse error in pattern validation (attempt {attempt + 1})", flush=True)
                        if attempt < max_retries - 1:
                            print(f"[{time_module.time() - start_time:.1f}s]    🔄 Retrying validation due to JSON error...", flush=True)
                            time_module.sleep(1)
                            continue
                        else:
                            # JSON parsing failed all attempts
                            pattern['llm_validation'] = {
                                'validated': True,  # Conservative approach
                                'confidence': 'unknown',
                                'reason': 'JSON parsing failed after retries'
                            }
                            return pattern, True

                # No JSON found in response
                print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ No JSON in LLM response (attempt {attempt + 1})", flush=True)
                if attempt < max_retries - 1:
                    print(f"[{time_module.time() - start_time:.1f}s]    🔄 Retrying validation...", flush=True)
                    time_module.sleep(1)
                    continue
                else:
                    pattern['llm_validation'] = {
                        'validated': True,  # Conservative approach
                        'confidence': 'unknown',
                        'reason': 'No valid JSON response from LLM'
                    }
                    return pattern, True

            except Exception as e:
                print(f"[{time_module.time() - start_time:.1f}s]    ⚠️ Pattern validation failed (attempt {attempt + 1}): {str(e)[:50]}...", flush=True)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[{time_module.time() - start_time:.1f}s]    🔄 Retrying in {wait_time}s...", flush=True)
                    time_module.sleep(wait_time)
                    continue
                else:
                    # All attempts failed
                    pattern['llm_validation'] = {
                        'validated': True,  # Conservative approach
                        'confidence': 'unknown',
                        'reason': f'Validation failed after {max_retries} attempts: {str(e)[:50]}'
                    }
                    return pattern, True

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_pattern = {}
        for pattern_idx, pattern in enumerate(patterns):
            future = executor.submit(validate_single_pattern, pattern, pattern_idx, len(patterns), llm_model, start_time)
            future_to_pattern[future] = pattern_idx

        # Process validation results
        processed_validations = 0
        validated_count = 0

        for future in as_completed(future_to_pattern):
            pattern_idx = future_to_pattern[future]
            processed_validations += 1

            try:
                pattern, is_valid = future.result()
                if is_valid:
                    validated_count += 1
                    validated_patterns.append(pattern)

                # Progress update
                if processed_validations % 10 == 0 or processed_validations == len(patterns):
                    print(f"[{time.time() - start_time:.1f}s] 📊 Pattern validation progress: {processed_validations}/{len(patterns)} processed, {validated_count} confirmed", flush=True)
                    sys.stdout.flush()

            except Exception as e:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed validation for pattern {pattern_idx + 1}: {str(e)[:50]}...", flush=True)
                # Conservative approach: include pattern if validation fails
                validated_patterns.append(patterns[pattern_idx])

    print(f"[{time.time() - start_time:.1f}s] [OK] Grammar pattern validation complete: {len(validated_patterns)}/{len(patterns)} patterns confirmed", flush=True)
    sys.stdout.flush()

    return validated_patterns


# ============================================================================
# Utility Functions
# ============================================================================

def parse_llm_json_response(response_str):
    """Parse JSON response from LLM with multiple fallback strategies."""
    json_str = None

    if '```json' in response_str and '```' in response_str:
        start = response_str.find('```json') + 7
        end = response_str.find('```', start)
        if end > start:
            json_str = response_str[start:end].strip()
    elif '```' in response_str:
        # Try generic code block
        start = response_str.find('```') + 3
        end = response_str.find('```', start)
        if end > start:
            json_str = response_str[start:end].strip()

    if json_str is None and '[' in response_str and ']' in response_str:
        start = response_str.find('[')
        end = response_str.rfind(']') + 1
        json_str = response_str[start:end]

    return json_str


# NOTE: Spelling error analysis functions have been removed.
# This script now focuses on grammar pattern analysis across 4 dimensions:
# D1-Structure Completeness, D2-Logical Connection, D3-Modification Preference, D4-Feature Bias


def call_llm_with_json_parsing(prompt, context="", start_time=0, max_retries=1):
    """Unified LLM call with JSON parsing and error handling."""
    import sys

    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = get_gm_model().invoke(messages)
            response_str = response.content.strip()

            # Parse JSON response
            json_str = parse_llm_json_response(response_str)
            if json_str:
                return json.loads(json_str)

            # If we get here, JSON parsing failed
            if attempt == max_retries - 1:
                handle_llm_error(response_str, context, start_time)
                return None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ LLM call failed in {context}: {str(e)[:50]}...", flush=True)
                sys.stdout.flush()
                return None

    return None


def log_message(message, start_time=0, level="INFO"):
    """Unified logging function with timestamp."""
    timestamp = f"[{time.time() - start_time:.1f}s]" if start_time > 0 else ""
    print(f"{timestamp} {message}", flush=True)
    sys.stdout.flush()

def realtime_print(message, start_time=0):
    """Real-time printing function that ensures immediate output."""
    timestamp = f"[{time.time() - start_time:.1f}s]" if start_time > 0 else ""
    print(f"{timestamp} {message}", flush=True)
    sys.stdout.flush()


# NOTE: print_error_details function removed - now using grammar pattern logging


# NOTE: Spelling error analysis functions have been replaced with grammar pattern analysis.
# The script now uses rule-based grammar pattern detection across 4 dimensions.


# NOTE: handle_llm_error function removed - now using grammar pattern analysis


# ============================================================================
# Main Processing Functions
# ============================================================================








# NOTE: analyze_single_sentence_second_pass function removed - replaced with grammar pattern analysis


# NOTE: analyze_single_sentence and tokenize_words functions removed - replaced with grammar pattern analysis


def validate_single_error(error, error_idx, total_errors, llm_model, start_time=0, max_retries=3):
    """Validate a single error using LLM with retry logic for JSON parsing errors."""
    import sys

    for attempt in range(max_retries):
        try:
            print(f"[{time.time() - start_time:.1f}s] 🔍 Validating error {error_idx + 1}/{total_errors}: '{error['word']}' → '{error['correct']}'" + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""), flush=True)
            sys.stdout.flush()

            # Create validation prompt
            validation_prompt = f"""<s> [INST] ## Task: Validate Spelling Error Classification

**Error Report to Validate:**
- Original word: "{error['word']}"
- Suggested correction: "{error['correct']}"
- Error category: "{error.get('error_category', 'Unknown')}"
- Error subcategory: "{error.get('error_subcategory', 'Unknown')}"
- Error explanation: "{error.get('error_explanation', 'Unknown')}"
- Context sentence: "{error['sentence']}"

**Validation Criteria:**
Determine if this error classification is accurate and fits our mandatory error classification system:

[OK] **SHOULD BE CONFIRMED** (return true) for properly classified errors that fit these categories:
- **Mechanical/Typo**: Random typing errors (Deletion, Insertion, Transposition, Scramble)
- **Phonetic**: Sound-based errors (Homophone, Suffix confusion)
- **Orthographic**: Knowledge-based errors (Hard Word spelling difficulties)

[FAIL] **SHOULD BE REJECTED** (return false) for:
- Incorrect or invalid error classifications
- Errors that do not fit any of the three main categories
- Grammar issues (subject-verb agreement, tense, prepositions, etc.)
- Spacing/punctuation/capitalization issues
- Style preferences or informal language choices
- Any corrections that are actually grammatical rather than spelling-based

**Decision Rules:**
- Verify that the error genuinely fits the assigned category and subcategory
- Be extremely conservative - only confirm errors with proper classification
- Amazon reviews are informal - respect colloquial language and domain terminology
- Beauty industry has specialized terms - do not flag legitimate domain vocabulary as errors

Return your decision as JSON:
{{
  "is_valid_error": true/false,
  "confidence_level": "high/medium/low",
  "validation_reason": "brief explanation of your decision"
}}
[/INST]"""

            messages = [{"role": "user", "content": validation_prompt}]
            response = llm_model.invoke(messages)
            response_str = response.content.strip()

            # Parse JSON response
            json_str = None
            if '```json' in response_str and '```' in response_str:
                start = response_str.find('```json') + 7
                end = response_str.find('```', start)
                if end > start:
                    json_str = response_str[start:end].strip()
            elif '{' in response_str and '}' in response_str:
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                json_str = response_str[start:end]

            if json_str:
                try:
                    validation_result = json.loads(json_str)
                    is_valid = validation_result.get('is_valid_error', False)
                    confidence = validation_result.get('confidence_level', 'unknown')
                    reason = validation_result.get('validation_reason', 'Unknown reason')

                    if is_valid:
                        print(f"[{time.time() - start_time:.1f}s]    [OK] Confirmed (confidence: {confidence})", flush=True)
                        return error, True
                    else:
                        word = error.get('word', 'unknown')
                        correct = error.get('correct', 'unknown')
                        print(f"[{time.time() - start_time:.1f}s]    [FAIL] Rejected (confidence: {confidence}) '{word}' → '{correct}' - {reason}", flush=True)
                        return error, False

                except json.JSONDecodeError as e:
                    print(f"[{time.time() - start_time:.1f}s]    ⚠️ JSON parse error in validation (attempt {attempt + 1}/{max_retries}): {str(e)}", flush=True)
                    if attempt < max_retries - 1:
                        print(f"[{time.time() - start_time:.1f}s]    🔄 Retrying validation request...", flush=True)
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        print(f"[{time.time() - start_time:.1f}s]    📄 Raw validation response (first 300 chars): {response_str[:300]}", flush=True)

                        # Additional diagnostics for validation
                        if '[INST]' in response_str:
                            print(f"[{time.time() - start_time:.1f}s]    🔍 DIAGNOSIS: Validation LLM returned the prompt itself", flush=True)
                        elif '{' not in response_str:
                            print(f"[{time.time() - start_time:.1f}s]    🔍 DIAGNOSIS: No JSON object found in validation response", flush=True)
                        else:
                            print(f"[{time.time() - start_time:.1f}s]    🔍 DIAGNOSIS: JSON structure issue in validation response", flush=True)

                        # If all retries failed, assume it's valid to be conservative
                        return error, True
            else:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ No JSON found in validation response (attempt {attempt + 1}/{max_retries})", flush=True)
                if attempt < max_retries - 1:
                    print(f"[{time.time() - start_time:.1f}s]    🔄 Retrying validation request...", flush=True)
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"[{time.time() - start_time:.1f}s]    📄 Raw validation response (first 200 chars): {response_str[:200]}", flush=True)
                    # If all retries failed, assume it's valid to be conservative
                    return error, True

        except Exception as e:
            print(f"[{time.time() - start_time:.1f}s]    ⚠️ Validation failed for error {error_idx + 1} (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}...", flush=True)
            if attempt < max_retries - 1:
                print(f"[{time.time() - start_time:.1f}s]    🔄 Retrying validation request...", flush=True)
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                # If all retries failed, assume it's valid to be conservative
                return error, True


def validate_reason_accuracy(error, llm_model, start_time=0):
    """Validate that the reason field accurately describes the actual error."""
    import sys

    try:
        # Create explanation validation prompt
        reason_validation_prompt = f'''<s> [INST] ## Task: Validate Error Classification Explanation

**Error Details:**
- Original word: "{error['word']}"
- Suggested correction: "{error['correct']}"
- Error category: "{error.get('error_category', 'Unknown')}"
- Error subcategory: "{error.get('error_subcategory', 'Unknown')}"
- Provided explanation: "{error.get('error_explanation', 'Unknown')}"
- Context sentence: "{error['sentence']}"

**Validation Task:**
Evaluate whether the provided explanation accurately describes why this error fits its assigned category and subcategory.

**Explanation Accuracy Criteria:**
[OK] **ACCURATE** if the explanation correctly describes:
- **Mechanical/Typo**: Random typing errors, keyboard issues, or input mistakes
- **Phonetic**: Sound-based retrieval errors or pronunciation confusions
- **Orthographic**: Knowledge gaps in spelling complex or domain-specific terms

[FAIL] **INACCURATE** if the explanation:
- Doesn't match the assigned error category/subcategory
- Describes a different type of error than what was classified
- Contradicts the actual error characteristics

**Response Format:**
{{
  "is_reason_accurate": true/false,
  "accuracy_confidence": "high/medium/low",
  "corrected_reason": "If inaccurate, provide the correct explanation",
  "explanation": "Brief explanation of the evaluation"
}}

[/INST]'''

        messages = [{"role": "user", "content": reason_validation_prompt}]
        response = llm_model.invoke(messages)
        response_str = response.content.strip()

        # Parse JSON response
        json_str = None
        if '```json' in response_str and '```' in response_str:
            start = response_str.find('```json') + 7
            end = response_str.find('```', start)
            if end > start:
                json_str = response_str[start:end].strip()
        elif '{' in response_str and '}' in response_str:
            start = response_str.find('{')
            end = response_str.rfind('}') + 1
            json_str = response_str[start:end]

        if json_str:
            try:
                validation_result = json.loads(json_str)
                is_accurate = validation_result.get('is_reason_accurate', True)
                confidence = validation_result.get('accuracy_confidence', 'unknown')
                corrected_reason = validation_result.get('corrected_reason', None)
                explanation = validation_result.get('explanation', 'Unknown')

                return is_accurate, confidence, corrected_reason, explanation

            except json.JSONDecodeError as e:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ JSON parse error in reason validation: {str(e)}", flush=True)
                return True, 'unknown', None, 'JSON parsing failed'

        return True, 'unknown', None, 'No JSON response'

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s]    ⚠️ Reason validation failed: {str(e)[:50]}...", flush=True)
        return True, 'unknown', None, 'Validation failed'


def validate_errors_with_llm(all_errors, llm_model, start_time=0):
    """Unified validation of detected errors and their explanations using LLM with concurrency."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    validated_errors = []

    if not all_errors:
        return validated_errors

    # Use reasonable concurrency for comprehensive validation
    max_validation_concurrent = min(80, len(all_errors))

    print(f"[{time.time() - start_time:.1f}s] 🔍 Starting comprehensive validation of {len(all_errors)} errors (max {max_validation_concurrent} concurrent)...", flush=True)
    sys.stdout.flush()

    def validate_single_error_comprehensive(error, error_idx, total_errors, llm_model, start_time):
        """Validate both error validity and explanation accuracy in one call."""
        # First validate error itself
        error, is_valid = validate_single_error(error, error_idx, total_errors, llm_model, start_time)

        if is_valid:
            # Then validate explanation accuracy
            try:
                is_accurate, confidence, corrected_reason, explanation = validate_reason_accuracy(error, llm_model, start_time)

                if not is_accurate and corrected_reason:
                    # Update error with corrected explanation
                    error['error_explanation'] = corrected_reason
                    error['original_explanation'] = error.get('error_explanation', '')

            except Exception:
                # If explanation validation fails, keep original
                pass

        return error, is_valid

    with ThreadPoolExecutor(max_workers=max_validation_concurrent) as executor:
        # Submit comprehensive validation tasks
        future_to_error = {}
        for error_idx, error in enumerate(all_errors):
            future = executor.submit(validate_single_error_comprehensive, error, error_idx, len(all_errors), llm_model, start_time)
            future_to_error[future] = error_idx

        # Process completed validations
        processed_validations = 0
        validated_count = 0
        explanation_corrections = 0

        for future in as_completed(future_to_error):
            error_idx = future_to_error[future]
            processed_validations += 1

            try:
                error, is_valid = future.result()
                if is_valid:
                    validated_count += 1
                    validated_errors.append(error)

                    # Check if explanation was corrected
                    if 'original_explanation' in error:
                        explanation_corrections += 1

                # Progress update
                if processed_validations % 8 == 0 or processed_validations == len(all_errors):
                    print(f"[{time.time() - start_time:.1f}s] 📊 Validation progress: {processed_validations}/{len(all_errors)} processed, {validated_count} confirmed, {explanation_corrections} explanations corrected", flush=True)
                    sys.stdout.flush()

            except Exception as e:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed validation for error {error_idx + 1}: {str(e)[:50]}...", flush=True)
                # Conservative approach: include error if validation fails
                validated_errors.append(all_errors[error_idx])

    print(f"[{time.time() - start_time:.1f}s] [OK] Comprehensive validation complete: {len(validated_errors)}/{len(all_errors)} errors confirmed, {explanation_corrections} explanations improved", flush=True)
    sys.stdout.flush()

    return validated_errors


def analyze_text_with_grammar_patterns(text, llm_model, start_time=0):
    """Analyze text for grammar patterns using LLM directly (no spaCy preprocessing)."""
    import sys

    # Preprocess text: replace hyphens with spaces to avoid tokenization issues
    text = text.replace("-", " ")
    print(f"[{time.time() - start_time:.1f}s] 📝 Preprocessed text (hyphens replaced with spaces)", flush=True)

    # Analyze grammar patterns using LLM directly
    print(f"[{time.time() - start_time:.1f}s] 🧠 Analyzing grammar patterns with LLM approach...", flush=True)
    sys.stdout.flush()

    all_patterns = analyze_grammar_patterns(text, llm_model)

    print(f"[{time.time() - start_time:.1f}s] 📊 Found {len(all_patterns)} grammar patterns", flush=True)

    # Group patterns by sentence for organized display
    patterns_by_sentence = {}
    for pattern in all_patterns:
        sent_idx = pattern['sentence_idx']
        sentence = pattern['sentence']
        key = f"{sent_idx}: {sentence}"
        if key not in patterns_by_sentence:
            patterns_by_sentence[key] = []
        patterns_by_sentence[key].append(pattern)

    # Print pattern details organized by sentence
    for sentence_key, sentence_patterns in patterns_by_sentence.items():
        sent_idx, sentence = sentence_key.split(': ', 1)
        print(f"\n{'='*80}", flush=True)
        print(f"📝 Sentence {sent_idx}: {sentence}", flush=True)
        print(f"🔍 Detected {len(sentence_patterns)} grammar pattern(s):", flush=True)
        print(f"{'-'*50}", flush=True)

        for pattern in sentence_patterns:
            dimension = pattern['dimension']
            pattern_name = pattern['pattern']
            confidence = pattern['confidence']
            evidence = pattern['evidence']

            confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
            print(f"  📋 {dimension}-{pattern_name} (confidence: {confidence_str})", flush=True)
            print(f"     Evidence: {evidence}", flush=True)
            print(f"     Time: [{time.time() - start_time:.1f}s]", flush=True)

        print(f"{'='*80}\n", flush=True)
        sys.stdout.flush()

    # Second stage: LLM validation of detected patterns using the same prompt
    if all_patterns and len(all_patterns) > 0:
        print(f"[{time.time() - start_time:.1f}s] 🔍 Starting second-stage LLM validation of {len(all_patterns)} detected patterns...", flush=True)
        sys.stdout.flush()

        validated_patterns = validate_grammar_patterns_by_sentence(all_patterns, llm_model, start_time)
        print(f"[{time.time() - start_time:.1f}s] [OK] Two-stage LLM analysis complete: {len(validated_patterns)} patterns detected and validated", flush=True)
        sys.stdout.flush()

        return validated_patterns

    print(f"[{time.time() - start_time:.1f}s] ⚠️ No grammar patterns found to validate", flush=True)
    sys.stdout.flush()
    return all_patterns




def process_users(input_file, output_file, llm_model, start_time=0):
    """Process users: find a user with 10+ reviews and send to LLM for analysis (optimized single pass)."""
    import sys
    import time
    print(f"[{time.time() - start_time:.1f}s] Loading data from: {input_file}", flush=True)
    sys.stdout.flush()

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"[{time.time() - start_time:.1f}s] [FAIL] Input file does not exist: {input_file}", flush=True)
        sys.stdout.flush()
        return None

    # Fast approach: find first user with 100+ reviews, then collect all their reviews
    print(f"[{time.time() - start_time:.1f}s] 🔍 Fast approach: finding first user with 100+ reviews...", flush=True)
    sys.stdout.flush()

    user_review_counts = {}
    target_user = None
    target_user_reviews = 0

    # PASS 1: Find the first user with 100+ reviews
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    review_data = json.loads(line)
                    user_id = review_data.get('user_id') or review_data.get('reviewerID') or review_data.get('reviewer_id')
                    if user_id:
                        user_review_counts[user_id] = user_review_counts.get(user_id, 0) + 1
                        current_count = user_review_counts[user_id]

                        # Check if this user has 100+ reviews - select the first user we find
                        if current_count >= 100 and target_user is None:
                            target_user = user_id
                            target_user_reviews = current_count
                            print(f"[{time.time() - start_time:.1f}s] 🎯 Found target user {target_user} with {target_user_reviews} reviews at line {line_num}", flush=True)
                            sys.stdout.flush()
                            # Stop scanning immediately once we find a user with 100+ reviews
                            break

                except json.JSONDecodeError:
                    continue

                # Show progress every 10000 lines
                if line_num % 10000 == 0:
                    print(f"🔄 Scanned {line_num} lines, found {len(user_review_counts)} unique users...", flush=True)
                    if target_user:
                        print(f"   📋 Target: {target_user} with {target_user_reviews} reviews so far", flush=True)
                    sys.stdout.flush()

        if not target_user:
            print(f"[{time.time() - start_time:.1f}s] [FAIL] No user found with 100+ reviews", flush=True)
            sys.stdout.flush()
            return None

        print(f"[{time.time() - start_time:.1f}s] 🎯 Selected first user with 100+ reviews: {target_user} with {target_user_reviews} reviews", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] [FAIL] Error in first pass: {e}", flush=True)
        sys.stdout.flush()
        return None

    # PASS 2: Collect all reviews for the target user
    print(f"[{time.time() - start_time:.1f}s] 📖 Second pass: collecting all reviews for user {target_user}...", flush=True)
    sys.stdout.flush()

    user_reviews_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    review_data = json.loads(line)
                    user_id = review_data.get('user_id') or review_data.get('reviewerID') or review_data.get('reviewer_id')

                    # Only collect reviews for our target user
                    if user_id == target_user:
                        text = review_data.get('reviewText') or review_data.get('text') or review_data.get('review_body', '')
                        if text.strip():
                            user_reviews_list.append(text.strip())

                except json.JSONDecodeError:
                    continue

        if not user_reviews_list:
            print(f"[{time.time() - start_time:.1f}s] [FAIL] No reviews found for user {target_user}", flush=True)
            sys.stdout.flush()
            return None

        print(f"[{time.time() - start_time:.1f}s] 📖 Collected {len(user_reviews_list)} reviews for user {target_user}", flush=True)

        # Combine all reviews for this user
        combined_text = ' '.join(user_reviews_list)
        print(f"[{time.time() - start_time:.1f}s] 📏 Combined text length: {len(combined_text)} characters", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] [FAIL] Error in second pass: {e}", flush=True)
        sys.stdout.flush()
        return None

    # Analyze grammar patterns
    print(f"[{time.time() - start_time:.1f}s] 🧠 Analyzing grammar patterns (all sentences)...", flush=True)
    sys.stdout.flush()

    try:
        grammar_patterns = analyze_text_with_grammar_patterns(combined_text, llm_model, start_time)
        print(f"[{time.time() - start_time:.1f}s] Grammar pattern analysis completed, found {len(grammar_patterns)} patterns", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] [FAIL] Grammar pattern analysis failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None

    # Summarize patterns by dimension
    dimension_summary = {}
    for pattern in grammar_patterns:
        dim = pattern['dimension']
        pat = pattern['pattern']
        if dim not in dimension_summary:
            dimension_summary[dim] = {}
        if pat not in dimension_summary[dim]:
            dimension_summary[dim][pat] = 0
        dimension_summary[dim][pat] += 1

    print(f"[{time.time() - start_time:.1f}s] 📊 Grammar pattern summary:", flush=True)
    for dim, patterns in dimension_summary.items():
        dim_name = GRAMMAR_DIMENSIONS[dim]['name']
        print(f"[{time.time() - start_time:.1f}s]   {dim} ({dim_name}): {patterns}", flush=True)
    sys.stdout.flush()

    # Save results
    result = {
        'user_id': target_user,
        'review_count': len(user_reviews_list),
        'text_length': len(combined_text),
        'grammar_patterns': grammar_patterns,
        'pattern_count': len(grammar_patterns),
        'dimension_summary': dimension_summary,
        'sample_reviews': user_reviews_list[:3],  # Save first 3 reviews as sample
        'grammar_dimensions': GRAMMAR_DIMENSIONS
    }

    try:
        print(f"[{time.time() - start_time:.1f}s] 💾 Saving results...", flush=True)
        sys.stdout.flush()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{time.time() - start_time:.1f}s] [OK] Results saved to: {output_file}", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] [FAIL] Failed to save results: {e}", flush=True)
        sys.stdout.flush()
        return None

    print(f"[{time.time() - start_time:.1f}s] 🎉 Found {len(grammar_patterns)} grammar patterns across {len(user_reviews_list)} reviews (all sentences)", flush=True)
    sys.stdout.flush()

    return result


def test_sentences():
    """Test basic spaCy functionality for grammar analysis."""
    import sys

    # Initialize spaCy for testing
    try:
        test_nlp = spacy.load("en_core_web_sm")
        print(f"[OK] spaCy model loaded successfully for testing", flush=True)
        sys.stdout.flush()
    except OSError as e:
        print(f"⚠️ spaCy model not available for testing (this is normal in some environments): {e}", flush=True)
        sys.stdout.flush()
        return
    except Exception as e:
        print(f"⚠️ Failed to load spaCy model for testing (this is normal in some environments): {e}", flush=True)
        sys.stdout.flush()
        return

    try:
        # Test basic tokenization
        test_text = "I want to find a cream for my skin."
        doc = test_nlp(test_text)

        print(f"Testing basic tokenization: '{test_text}'", flush=True)
        print(f"Found {len(doc)} tokens", flush=True)
        sys.stdout.flush()

        # Test POS tagging
        verbs = [token for token in doc if token.pos_.startswith('V')]
        nouns = [token for token in doc if token.pos_.startswith('N')]

        print(f"Verbs: {[v.text for v in verbs]}", flush=True)
        print(f"Nouns: {[n.text for n in nouns]}", flush=True)
        sys.stdout.flush()

        print(f"[OK] Grammar analysis components working correctly", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"⚠️ spaCy testing failed, but this may not affect main functionality: {e}", flush=True)
        sys.stdout.flush()


def main():
    """Main function - Now only performs adversarial user fingerprint extraction."""
    import sys
    import time
    import json

    start_time = time.time()
    print(f"🚀 [{time.time() - start_time:.1f}s] Starting adversarial user fingerprint extraction...", flush=True)
    sys.stdout.flush()

    print(f"📍 [{time.time() - start_time:.1f}s] Initializing SiliconFlow model...", flush=True)
    sys.stdout.flush()

    try:
        llm_model = get_gm_model()
        print(f"[OK] [{time.time() - start_time:.1f}s] SiliconFlow model initialized successfully", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[FAIL] [{time.time() - start_time:.1f}s] Failed to initialize model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # Use fixed target user
    target_user = "AEZP6Z2C5AVQDZAJECQYZWQRNG3Q"
    print(f"👤 [{time.time() - start_time:.1f}s] Using fixed target user: {target_user}", flush=True)
    sys.stdout.flush()

    # Load user reviews for fingerprinting
    print(f"📁 [{time.time() - start_time:.1f}s] Loading user review data for {target_user}...", flush=True)
    sys.stdout.flush()

    try:
        # Load the same user data as before for consistency
        user_reviews = load_user_reviews_for_fingerprinting(target_user)
        print(f"[OK] [{time.time() - start_time:.1f}s] Loaded {len(user_reviews)} user reviews", flush=True)
        sys.stdout.flush()

        # Extract adversarial fingerprint
        print(f"🔍 [{time.time() - start_time:.1f}s] Extracting adversarial user fingerprint...", flush=True)
        sys.stdout.flush()

        fingerprint_result = extract_adversarial_user_fingerprint(user_reviews, llm_model, target_user)

        if fingerprint_result:
            try:
                # fingerprint_result is already a dict object from aggregate_sentence_analyses
                fingerprint_data = fingerprint_result
                print(f"[OK] [{time.time() - start_time:.1f}s] Fingerprint extraction successful!", flush=True)

                # Save fingerprint to file (keeping original filename)
                fingerprint_output = "/home/wlia0047/ar57_scratch/wenyu/amazon_review_grammer_analysis.json"
                with open(fingerprint_output, 'w', encoding='utf-8') as f:
                    json.dump(fingerprint_data, f, ensure_ascii=False, indent=2)

                print(f"💾 [{time.time() - start_time:.1f}s] Adversarial fingerprint saved to: {fingerprint_output}", flush=True)

                # Display comprehensive 8-dimension findings
                fingerprint = fingerprint_data.get('fingerprint', {})
                print(f"\n🎯 Comprehensive 8-Dimension Adversarial Fingerprint Findings:")

                # Syntactic features
                syntactic = fingerprint.get('syntactic', {})
                print(f"\n📊 Syntactic Structure:")

                # Padding phrases with frequencies
                padding_phrases = syntactic.get('padding_phrases', [])
                if padding_phrases:
                    print(f"   📝 Padding phrases:")
                    for item in padding_phrases:
                        phrase = item.get('phrase', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      • '{phrase}' ({freq} times)")

                # Interruption types (top 3)
                int_types = syntactic.get('interruption_types', [])
                if int_types:
                    print(f"   🔀 Interruption types:")
                    for i, item in enumerate(int_types[:3], 1):
                        typ = item.get('type', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      {i}. {typ} ({freq} times)")

                # Conditional usage levels (top 3)
                cond_levels = syntactic.get('conditional_usage_levels', [])
                if cond_levels:
                    print(f"   ❓ Conditional usage levels:")
                    for i, item in enumerate(cond_levels[:3], 1):
                        level = item.get('level', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      {i}. {level} ({freq} times)")

                # Lexical features
                lexical = fingerprint.get('lexical', {})
                print(f"\n📝 Lexical Choice:")

                # Circumlocution habits (top 3)
                circ_habits = lexical.get('circumlocution_habits', [])
                if circ_habits:
                    print(f"   🔄 Circumlocution habits:")
                    for i, item in enumerate(circ_habits[:3], 1):
                        typ = item.get('type', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      {i}. {typ} ({freq} times)")

                # Generic nouns with frequencies
                generic_nouns = lexical.get('preferred_generic_nouns', [])
                if generic_nouns:
                    print(f"   📦 Generic nouns:")
                    for item in generic_nouns:
                        noun = item.get('noun', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      • '{noun}' ({freq} times)")

                # Logic features
                logic = fingerprint.get('logic', {})
                print(f"\n🧠 Logic & Context:")

                # Past reference habits (top 3)
                past_habits = logic.get('past_reference_habits', [])
                if past_habits:
                    print(f"   ⏰ Past reference habits:")
                    for i, item in enumerate(past_habits[:3], 1):
                        level = item.get('level', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      {i}. {level} ({freq} times)")

                # Negation styles (top 3)
                neg_styles = logic.get('negation_styles', [])
                if neg_styles:
                    print(f"   ⚡ Negation styles:")
                    for i, item in enumerate(neg_styles[:3], 1):
                        typ = item.get('type', 'Unknown')
                        freq = item.get('frequency', 0)
                        print(f"      {i}. {typ} ({freq} times)")

                # Contextual insights
                contextual_insights = fingerprint.get('contextual_insights', {})
                if contextual_insights:
                    print(f"\n🔍 Contextual Insights:")

                    syntactic_insights = contextual_insights.get('syntactic', {})
                    if syntactic_insights:
                        print(f"   Syntactic contexts:")
                        print(f"     • Padding phrases: {syntactic_insights.get('padding_when', 'Unknown')}")
                        print(f"     • Interruptions: {syntactic_insights.get('interruption_when', 'Unknown')}")
                        print(f"     • Conditionals: {syntactic_insights.get('conditional_when', 'Unknown')}")

                    lexical_insights = contextual_insights.get('lexical', {})
                    if lexical_insights:
                        print(f"   Lexical contexts:")
                        print(f"     • Circumlocution: {lexical_insights.get('circumlocution_when', 'Unknown')}")
                        print(f"     • Generic nouns: {lexical_insights.get('generic_nouns_when', 'Unknown')}")

                    logic_insights = contextual_insights.get('logic', {})
                    if logic_insights:
                        print(f"   Logic contexts:")
                        print(f"     • Past references: {logic_insights.get('past_reference_when', 'Unknown')}")
                        print(f"     • Negation styles: {logic_insights.get('negation_when', 'Unknown')}")

                # Summary and percentages
                print(f"\n📋 Style Summary: {fingerprint.get('style_summary', 'No summary available')}")

                # Dimension percentages
                analysis_summary = fingerprint.get('analysis_summary', {})
                percentages = analysis_summary.get('dimension_percentages', {})

                if percentages:
                    print(f"\n📊 Dimension Coverage (Decimal):")
                    print(f"   🔤 Padding phrases: {percentages.get('padding_phrases', 0) / 100:.3f}")
                    print(f"   🔀 Interruption types: {percentages.get('interruption_types', 0) / 100:.3f}")
                    print(f"   ❓ Conditional usage: {percentages.get('conditional_usage', 0) / 100:.3f}")
                    print(f"   🔄 Circumlocution habits: {percentages.get('circumlocution_habits', 0) / 100:.3f}")
                    print(f"   📦 Generic nouns: {percentages.get('generic_nouns', 0) / 100:.3f}")
                    print(f"   ⏰ Past reference habits: {percentages.get('past_reference_habits', 0) / 100:.3f}")
                    print(f"   ⚡ Negation styles: {percentages.get('negation_styles', 0) / 100:.3f}")

            except json.JSONDecodeError as e:
                print(f"[FAIL] [{time.time() - start_time:.1f}s] Failed to parse fingerprint JSON: {e}", flush=True)
                print(f"Raw response: {fingerprint_result[:500]}...", flush=True)
                return None
        else:
            print(f"[FAIL] [{time.time() - start_time:.1f}s] Fingerprint extraction failed", flush=True)
            return None

    except Exception as e:
        print(f"[FAIL] [{time.time() - start_time:.1f}s] Error during fingerprinting: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None

    print(f"🏁 [{time.time() - start_time:.1f}s] Adversarial user fingerprint extraction completed successfully!", flush=True)
    sys.stdout.flush()

    return fingerprint_data


def find_target_user():
    """Find the first user with 100+ reviews and return their user ID."""
    import json
    import os
    import sys

    user_review_counts = {}
    target_user = None

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    review_data = json.loads(line)
                    user_id = review_data.get('user_id') or review_data.get('reviewerID') or review_data.get('reviewer_id')
                    if user_id:
                        user_review_counts[user_id] = user_review_counts.get(user_id, 0) + 1
                        current_count = user_review_counts[user_id]

                        # Check if this user has 100+ reviews - select the first user we find
                        if current_count >= 100 and target_user is None:
                            target_user = user_id
                            print(f"🎯 Found target user {target_user} with {current_count} reviews at line {line_num}", flush=True)
                            break

                except json.JSONDecodeError:
                    continue

                # Show progress every 10000 lines
                if line_num % 10000 == 0:
                    print(f"🔄 Scanned {line_num} lines, found {len(user_review_counts)} unique users...", flush=True)
                    sys.stdout.flush()

        if not target_user:
            print(f"[FAIL] No user found with 100+ reviews", flush=True)
            return None

        print(f"🎯 Selected first user with 100+ reviews: {target_user}", flush=True)
        return target_user

    except Exception as e:
        print(f"Error finding target user: {e}")
        return None


def load_user_reviews_for_fingerprinting(target_user):
    """Load user reviews for fingerprinting analysis."""
    import json

    user_reviews = []

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    review_data = json.loads(line)
                    user_id = review_data.get('user_id') or review_data.get('reviewerID') or review_data.get('reviewer_id')

                    # Focus on the target user
                    if user_id == target_user:
                        text = review_data.get('reviewText') or review_data.get('text') or review_data.get('review_body', '')
                        if text.strip():
                            user_reviews.append(text.strip())

                except json.JSONDecodeError:
                    continue

                # Limit to reasonable number for fingerprinting
                if len(user_reviews) >= 100:  # Get more reviews for better fingerprinting
                    break

    except Exception as e:
        print(f"Error loading user reviews: {e}")
        return []

    return user_reviews


# ============================================================================
# COMMENTED OUT: Previous Grammar Pattern Analysis Functions
# ============================================================================
# The following functions have been commented out as the script now focuses
# solely on adversarial user fingerprint extraction.
#
# - analyze_grammar_patterns()
# - analyze_single_sentence_llm()
# - validate_sentence_patterns()
# - validate_grammar_patterns_by_sentence()
# - analyze_text_with_grammar_patterns()
# - process_users()
# ============================================================================

def extract_adversarial_user_fingerprint(user_reviews, llm_model, real_user_id=None):
    """
    Analyzes user reviews to extract specific vocabulary and habits
    for the 4 adversarial grammar dimensions.
    Processes ALL reviews, sentence by sentence.
    """
    import time

    print(f"📝 Starting sentence-by-sentence analysis of {len(user_reviews)} reviews...", flush=True)

    # Step 1: Combine all reviews and split into sentences
    combined_text = ' '.join(user_reviews)
    print(f"📄 Combined text length: {len(combined_text)} characters", flush=True)

    # Use NLTK to split into sentences
    try:
        sentences = sent_tokenize(combined_text)
        print(f"✂️  Split into {len(sentences)} sentences", flush=True)
    except (ImportError, LookupError) as e:
        print(f"[FAIL] NLTK is not available or punkt tokenizer not found: {e}", flush=True)
        print("Please install NLTK and download punkt tokenizer: pip install nltk && python -c \"import nltk; nltk.download('punkt')\"", flush=True)
        sys.exit(1)

    # Step 2: Analyze each sentence individually with concurrency
    # Filter sentences that need analysis (more than 5 words)
    sentences_to_analyze = []
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count > 5:
            sentences_to_analyze.append(sentence)

    total_sentences_to_analyze = len(sentences_to_analyze)
    print(f"📋 Found {total_sentences_to_analyze} sentences to analyze (skipped {len(sentences) - total_sentences_to_analyze} short sentences)", flush=True)

    if total_sentences_to_analyze == 0:
        print(f"[FAIL] No sentences long enough for analysis (all sentences have 5 words or fewer)")
        print("Cannot generate user fingerprint without sentence analyses.")
        sys.exit(1)

    # Concurrent analysis with 200 workers
    max_concurrent = min(200, total_sentences_to_analyze)
    print(f"🚀 Starting concurrent analysis with {max_concurrent} workers...", flush=True)

    sentence_analyses = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all analysis tasks
        future_to_sentence = {
            executor.submit(analyze_single_sentence, sentence, llm_model): sentence
            for sentence in sentences_to_analyze
        }

        # Process completed tasks
        completed_count = 0
        for future in as_completed(future_to_sentence):
            sentence = future_to_sentence[future]
            completed_count += 1

            try:
                analysis_result = future.result()
                if analysis_result:
                    sentence_analyses.append({
                        'sentence': sentence,
                        'analysis': analysis_result
                    })
                else:
                    # Analysis returned None (likely LLM failure)
                    print(f"⚠️  Sentence analysis returned no result: {sentence[:50]}...", flush=True)

                # Progress reporting
                if completed_count % 50 == 0 or completed_count == total_sentences_to_analyze:
                    success_count = len(sentence_analyses)
                    print(f"📊 Completed {completed_count}/{total_sentences_to_analyze} sentence analyses ({success_count} successful)", flush=True)

            except Exception as e:
                print(f"❌ Sentence analysis task failed for: {sentence[:50]}... Error: {e}")
                # Continue processing other sentences even if one fails

    print(f"✅ Completed concurrent analysis of {len(sentence_analyses)} sentences", flush=True)

    # Check if we have any successful analyses
    if len(sentence_analyses) == 0:
        print(f"[FAIL] All {total_sentences_to_analyze} sentence analyses failed. LLM service may be unavailable.")
        print("Cannot generate user fingerprint without any sentence analyses.")
        sys.exit(1)

    # Step 3: Aggregate results across all sentences
    return aggregate_sentence_analyses(sentence_analyses, real_user_id)


def analyze_single_sentence(sentence, llm_model):
    """Analyze a single sentence for comprehensive grammar patterns across 8 dimensions."""
    sentence_prompt = f"""<s> [INST] ## Task: Analyze Grammar Patterns with Context

Analyze this single sentence for writing habits and patterns across 7 adversarial dimensions. For each identified pattern, explain WHEN and WHY the user tends to use it.

**Sentence:** "{sentence}"

**Analysis Dimensions:**

### Group A: Syntactic Structure (Sentence Level)
1. **Padding/Fluff**: Any filler words, hesitation markers, or introductory phrases? (e.g., "To be honest", "Basically")
   - **Context**: When do they use these? (expressing doubt, softening criticism, building rapport)
2. **Interruption Style**: Do they use dashes, parenthesis, or comma clauses to insert stories? What kind of stories?
   - **Context**: When do they interrupt? (adding personal anecdotes, technical details, emotional reactions)
3. **Conditionals**: Do they use complex "If...then..." or "Provided that..." structures?
   - **Context**: When do they use conditions? (hypothetical scenarios, requirements, uncertainty)

### Group B: Lexical Choice (Word Level)
4. **Circumlocution**: Do they describe *what an object does* instead of *what it is*? (e.g., "the thing for my eyes" vs "mascara")
   - **Context**: When do they avoid direct names? (uncertainty, politeness, lack of knowledge)
5. **Generic Nouns**: Preferred vague words (gadget, unit, thingy, stuff)?
   - **Context**: When do they use generic terms? (generalizing, avoiding specificity, casual communication)

### Group C: Logic & Context (Semantic Level)
6. **Temporal/Comparative**: Do they mention "Old products" or "Previous brands" to describe what they want now?
   - **Context**: When do they reference past experiences? (dissatisfaction, comparison, evolution)
7. **Negation Style**: Direct negation, indirect (not exactly, far from), or Litotes (not bad)?
   - **Context**: When do they use different negation styles? (softening criticism, politeness, emphasis)

**Output Format (JSON):**
{{
  "syntactic": {{
    "padding_phrases": ["phrase1", "phrase2"],
    "padding_context": "Used when expressing honest opinions or softening criticism",
    "interruption_type": "Narrative/Technical/Emotional/None",
    "interruption_context": "Used to add personal stories or technical details",
    "conditional_usage": "High/Low/None",
    "conditional_context": "Used for hypothetical scenarios or requirements"
  }},
  "lexical": {{
    "circumlocution_habit": "Function/Appearance/None",
    "circumlocution_context": "Used when uncertain about specific terms or being polite",
    "preferred_generic_nouns": ["stuff", "items"],
    "nouns_context": "Used in casual communication or when generalizing"
  }},
  "logic": {{
    "past_reference_habit": "High/Low/None",
    "past_reference_context": "Used when comparing old vs new products or expressing dissatisfaction",
    "negation_style": "Direct/Litotes/Exclusion/None",
    "negation_context": "Used to soften criticism or express disappointment politely"
  }}
}}
[/INST]"""

    try:
        messages = [{"role": "user", "content": sentence_prompt}]
        response = llm_model.invoke(messages)
        json_str = parse_llm_json_response(response.content)

        if json_str:
            # Parse the JSON string into a Python object
            import json
            return json.loads(json_str)
        else:
            return None
    except Exception as e:
        # Don't exit program in concurrent mode - just return None for this sentence
        # The concurrent processing loop will handle the failure
        return None


def generate_context_summaries(contexts_dict):
    """Generate contextual insights summaries from collected contexts, excluding 'None' and default values."""
    summaries = {}

    # Helper function to get most common meaningful context (excluding None/defaults)
    def get_meaningful_context(context_list):
        if not context_list:
            return "No meaningful contexts found - patterns may be rare or subtle"

        # Filter out meaningless contexts
        meaningful_contexts = []
        meaningless_patterns = [
            "no", "none", "not", "n/a", "unknown", "context not available",
            "no filler phrases", "no interruption", "no conditional",
            "no generic nouns", "no negation", "no temporal"
        ]

        for ctx in context_list:
            ctx_lower = ctx.lower().strip()
            # Skip if it contains meaningless patterns
            if not any(pattern in ctx_lower for pattern in meaningless_patterns):
                meaningful_contexts.append(ctx)

        if not meaningful_contexts:
            return "No meaningful contexts found - patterns may be rare or subtle"

        # Return the most common meaningful context
        context_freq = {}
        for ctx in meaningful_contexts:
            context_freq[ctx] = context_freq.get(ctx, 0) + 1
        return max(context_freq.items(), key=lambda x: x[1])[0]

    summaries['syntactic'] = {
        'padding_when': get_meaningful_context(contexts_dict.get('padding_contexts', [])),
        'interruption_when': get_meaningful_context(contexts_dict.get('interruption_contexts', [])),
        'conditional_when': get_meaningful_context(contexts_dict.get('conditional_contexts', []))
    }

    summaries['lexical'] = {
        'circumlocution_when': get_meaningful_context(contexts_dict.get('circumlocution_contexts', [])),
        'generic_nouns_when': get_meaningful_context(contexts_dict.get('nouns_contexts', []))
    }

    summaries['logic'] = {
        'past_reference_when': get_meaningful_context(contexts_dict.get('past_reference_contexts', [])),
        'negation_when': get_meaningful_context(contexts_dict.get('negation_contexts', []))
    }

    return summaries


def aggregate_sentence_analyses(sentence_analyses, real_user_id):
    """Aggregate analysis results from all sentences into a comprehensive user fingerprint with 8 dimensions."""
    if not sentence_analyses:
        print(f"[FAIL] No valid sentence analyses available. All sentences were too short or LLM failed.")
        print("Cannot generate user fingerprint without sentence analyses.")
        sys.exit(1)

    print(f"🔄 Aggregating results from {len(sentence_analyses)} sentence analyses...", flush=True)

    # Initialize aggregation structures for 7 dimensions with context
    # Group A: Syntactic Structure
    padding_phrases = {}
    padding_contexts = []
    interruption_types = {}
    interruption_contexts = []
    conditional_usage = {}
    conditional_contexts = []

    # Group B: Lexical Choice
    circumlocution_habits = {}
    circumlocution_contexts = []
    generic_nouns = {}
    nouns_contexts = []

    # Group C: Logic & Context
    past_reference_habits = {}
    past_reference_contexts = []
    negation_styles = {}
    negation_contexts = []

    # Count frequencies across all analyses
    for analysis_data in sentence_analyses:
        analysis = analysis_data.get('analysis', {})

        # Extract syntactic features
        syntactic = analysis.get('syntactic', {})
        for phrase in syntactic.get('padding_phrases', []):
            padding_phrases[phrase] = padding_phrases.get(phrase, 0) + 1

        int_type = syntactic.get('interruption_type', 'None')
        interruption_types[int_type] = interruption_types.get(int_type, 0) + 1

        # Collect context information ONLY for sentences that actually exhibit the patterns
        if syntactic.get('padding_phrases') and len(syntactic.get('padding_phrases', [])) > 0:
            if syntactic.get('padding_context'):
                padding_contexts.append(syntactic['padding_context'])

        if int_type != 'None' and syntactic.get('interruption_context'):
            interruption_contexts.append(syntactic['interruption_context'])

        cond_usage = syntactic.get('conditional_usage', 'None')
        conditional_usage[cond_usage] = conditional_usage.get(cond_usage, 0) + 1
        if cond_usage != 'None' and syntactic.get('conditional_context'):
            conditional_contexts.append(syntactic['conditional_context'])

        # Extract lexical features
        lexical = analysis.get('lexical', {})
        circ_habit = lexical.get('circumlocution_habit', 'None')
        circumlocution_habits[circ_habit] = circumlocution_habits.get(circ_habit, 0) + 1
        if circ_habit != 'None' and lexical.get('circumlocution_context'):
            circumlocution_contexts.append(lexical['circumlocution_context'])

        for noun in lexical.get('preferred_generic_nouns', []):
            generic_nouns[noun] = generic_nouns.get(noun, 0) + 1
        if lexical.get('preferred_generic_nouns') and len(lexical.get('preferred_generic_nouns', [])) > 0:
            if lexical.get('nouns_context'):
                nouns_contexts.append(lexical['nouns_context'])

        # Extract logic features
        logic = analysis.get('logic', {})
        past_ref = logic.get('past_reference_habit', 'None')
        past_reference_habits[past_ref] = past_reference_habits.get(past_ref, 0) + 1
        if past_ref != 'None' and logic.get('past_reference_context'):
            past_reference_contexts.append(logic['past_reference_context'])

        neg_style = logic.get('negation_style', 'None')
        negation_styles[neg_style] = negation_styles.get(neg_style, 0) + 1
        if neg_style != 'None' and logic.get('negation_context'):
            negation_contexts.append(logic['negation_context'])

    # Select top results for each dimension with frequencies
    top_padding = sorted(padding_phrases.items(), key=lambda x: x[1], reverse=True)[:5]
    top_padding_with_freq = [{"phrase": phrase, "frequency": count} for phrase, count in top_padding]

    # Get top 3 interruption types
    top_interruption_items = sorted(interruption_types.items(), key=lambda x: x[1], reverse=True)[:3]
    top_interruption_list = [{"type": typ, "frequency": freq} for typ, freq in top_interruption_items]

    # Get top 3 conditional usage levels
    top_conditional_items = sorted(conditional_usage.items(), key=lambda x: x[1], reverse=True)[:3]
    top_conditional_list = [{"level": level, "frequency": freq} for level, freq in top_conditional_items]

    # Get top 3 circumlocution habits
    top_circumlocution_items = sorted(circumlocution_habits.items(), key=lambda x: x[1], reverse=True)[:3]
    top_circumlocution_list = [{"type": typ, "frequency": freq} for typ, freq in top_circumlocution_items]

    top_generic_nouns = sorted(generic_nouns.items(), key=lambda x: x[1], reverse=True)[:6]
    top_generic_nouns_with_freq = [{"noun": noun, "frequency": count} for noun, count in top_generic_nouns]

    # Get top 3 past reference habits
    top_past_ref_items = sorted(past_reference_habits.items(), key=lambda x: x[1], reverse=True)[:3]
    top_past_ref_list = [{"level": level, "frequency": freq} for level, freq in top_past_ref_items]

    # Get top 3 negation styles
    top_negation_items = sorted(negation_styles.items(), key=lambda x: x[1], reverse=True)[:3]
    top_negation_list = [{"type": typ, "frequency": freq} for typ, freq in top_negation_items]

    # Generate context summaries
    context_summaries = generate_context_summaries({
        'padding_contexts': padding_contexts,
        'interruption_contexts': interruption_contexts,
        'conditional_contexts': conditional_contexts,
        'circumlocution_contexts': circumlocution_contexts,
        'nouns_contexts': nouns_contexts,
        'past_reference_contexts': past_reference_contexts,
        'negation_contexts': negation_contexts
    })

    # Create comprehensive fingerprint with 7 dimensions, frequencies, and contextual insights
    aggregated_fingerprint = {
        "user_id": real_user_id if real_user_id else 'unknown_user',
        "fingerprint": {
            "syntactic": {
                "padding_phrases": top_padding_with_freq,
                "interruption_types": top_interruption_list,
                "conditional_usage_levels": top_conditional_list
            },
            "lexical": {
                "circumlocution_habits": top_circumlocution_list,
                "preferred_generic_nouns": top_generic_nouns_with_freq
            },
            "logic": {
                "past_reference_habits": top_past_ref_list,
                "negation_styles": top_negation_list
            },
            "contextual_insights": context_summaries,
            "style_summary": f"This user shows {top_circumlocution_list[0]['type'].lower() if top_circumlocution_list else 'unknown'} tendencies with {top_interruption_list[0]['type'].lower() if top_interruption_list else 'unknown'} interruptions and {top_negation_list[0]['type'].lower() if top_negation_list else 'unknown'} negation patterns.",
            "analysis_summary": {
                "total_sentences_analyzed": len(sentence_analyses),
                "dimension_percentages": {
                    "padding_phrases": len(padding_phrases) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "interruption_types": len(interruption_types) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "conditional_usage": len(conditional_usage) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "circumlocution_habits": len(circumlocution_habits) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "generic_nouns": len(generic_nouns) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "past_reference_habits": len(past_reference_habits) / len(sentence_analyses) * 100 if sentence_analyses else 0,
                    "negation_styles": len(negation_styles) / len(sentence_analyses) * 100 if sentence_analyses else 0
                },
                "raw_counts": {
                    "padding_phrases_found": len(padding_phrases),
                    "interruption_types_found": len(interruption_types),
                    "conditional_patterns_found": len(conditional_usage),
                    "circumlocution_habits_found": len(circumlocution_habits),
                    "generic_nouns_found": len(generic_nouns),
                    "past_reference_habits_found": len(past_reference_habits),
                    "negation_styles_found": len(negation_styles)
                }
            }
        }
    }

    print(f"✅ Comprehensive 8-dimension fingerprint aggregation complete", flush=True)
    return aggregated_fingerprint


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Amazon Review Adversarial User Fingerprint Extraction")
        print("Usage: python3 amazon_review_style_grammer_analysis.py")
        print("This script extracts adversarial grammar fingerprints from Amazon user reviews.")
        sys.exit(0)
    main()

