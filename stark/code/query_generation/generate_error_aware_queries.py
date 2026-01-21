#!/usr/bin/env python3
"""
Error-Aware Query Generation Script

This script:
1. Reads the output JSON from amazon_review_style_analysis.py
2. Analyzes user error patterns from the spelling errors
3. Generates a clean/original query about the product category (without user errors)
4. Creates a modified query that incorporates the user's error patterns while maintaining semantic meaning

Input: JSON file with user spelling error analysis
Output: JSON file with original and error-modified queries
"""

import os
import json
import sys
import time
from collections import Counter, defaultdict

# Ensure stark/code is on Python path so we can import model.py
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

# Import the model - required for this script
try:
    from model import get_gm_model
except ImportError as e:
    print(f"❌ Model import failed: {e}", flush=True)
    print(f"💥 This script requires the LLM model to function. Exiting.", flush=True)
    sys.exit(1)

# Configuration
INPUT_ANALYSIS_FILE = "/home/wlia0047/ar57_scratch/wenyu/style_analysis_first_user.json"
OUTPUT_QUERIES_FILE = "/home/wlia0047/ar57_scratch/wenyu/error_aware_queries.json"


def assess_word_difficulty(word):
    """Assess word difficulty based on multiple scientific factors."""
    word = word.lower().strip()

    # L_w: Word length (Length)
    L_w = len(word)

    # F_w: Frequency Score (inverse frequency - higher for rare words)
    # Using a simple corpus-based approach
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'with', 'hair', 'skin', 'good', 'like', 'love', 'best', 'make', 'free', 'natural', 'water', 'daily', 'great', 'nice', 'soft', 'fine', 'full', 'long', 'high', 'low', 'deep', 'light', 'dark', 'hot', 'cold', 'looking', 'shampoo', 'conditioner'}
    technical_beauty = {'hyaluronic', 'moisturizer', 'sulfate', 'keratin', 'peptide', 'vitamin', 'protein', 'antioxidant', 'regimen', 'emollient', 'humectant', 'occlusive', 'surfactant', 'curly', 'frizz', 'volume', 'thickness', 'weighing', 'argan', 'define', 'reduce'}

    if word in common_words:
        F_w = 2.0  # Frequent words are easier
    elif word in technical_beauty:
        F_w = 12.0  # Technical terms are much harder
    else:
        F_w = 6.0  # Neutral words

    # P_w: Phonetic Complexity (syllables and consonant clusters)
    vowels = 'aeiou'
    syllable_count = sum(1 for char in word if char in vowels)
    consonant_clusters = sum(1 for i in range(len(word)-1) if word[i] not in vowels and word[i+1] not in vowels)

    # Normalize phonetic complexity
    P_w = (syllable_count * 0.6) + (consonant_clusters * 0.4)

    # Overall difficulty score using the formula: D_w = (L_w * F_w * P_w) / 100
    # This creates a multiplicative effect where rare, long, complex words get very high difficulty
    D_w = (L_w * F_w * P_w) / 100.0

    return {
        'difficulty_score': round(D_w, 2),
        'length': L_w,
        'frequency_score': F_w,
        'phonetic_complexity': round(P_w, 2),
        'difficulty_level': 'low' if D_w < 3.0 else 'medium' if D_w < 6.0 else 'high'
    }


def assess_user_dna(errors):
    """Assess user's error DNA from their error history."""
    if not errors:
        return {
            'base_error_rate': 0.0,
            'error_type_vector': {},
            'total_errors': 0
        }

    total_errors = len(errors)

    # E_u: Base error rate (errors per 100 words)
    # Assuming average review length ~100 words, calculate error rate
    E_u = (total_errors / 100.0) * 100  # Convert to percentage

    # T_u: Error type vector (user's classification tendencies)
    from collections import Counter
    error_counts = Counter()

    for error in errors:
        category = error.get('error_category', 'Unknown')
        subcategory = error.get('error_subcategory', 'Unknown')
        key = f"{category}/{subcategory}"
        error_counts[key] += 1

    T_u = {}
    for error_type, count in error_counts.items():
        T_u[error_type] = count / total_errors

    return {
        'base_error_rate': round(E_u, 2),
        'error_type_vector': T_u,
        'total_errors': total_errors
    }


def calculate_error_injection_probability(word_difficulty, user_dna, environment_factor=1.0):
    """Calculate dynamic error injection probability using the scientific formula."""

    D_w = word_difficulty['difficulty_score']  # Word difficulty
    E_u = user_dna['base_error_rate'] / 100.0  # Convert percentage to decimal

    # Core formula: Base weight scaled by user habits
    # P_base = D_w * E_u * α, where α is scaling coefficient (suggested range 0.5-2.0)
    α = 1.2  # Scaling coefficient
    P_base = D_w * E_u * α

    # Environment coefficient (mobile vs desktop)
    E_env = environment_factor

    # Sigmoid function to create S-curve: P_inject = 1 / (1 + e^(-k*(P_base * E_env - threshold)))
    k = 2.0  # Steepness parameter
    threshold = 1.5  # Threshold where error probability starts increasing rapidly

    exponent = -k * (P_base * E_env - threshold)
    P_inject = 1.0 / (1.0 + 2.71828 ** exponent)  # Using e≈2.71828

    return round(P_inject, 4)


def calculate_dynamic_error_type_probabilities(user_dna, word_difficulty):
    """Calculate error type probabilities adjusted by word difficulty."""

    D_w = word_difficulty['difficulty_score']
    T_u = user_dna['error_type_vector']

    # Difficulty-based adjustment operators
    difficulty_operators = {}

    if D_w < 3.0:  # Simple words
        difficulty_operators = {
            'Phonetic/Homophone': 0.3,      # Much less likely
            'Phonetic/Suffix': 0.3,
            'Orthographic/Hard Word': 0.2,  # Very unlikely
            'Mechanical/Typo/Deletion': 2.0, # Much more likely
            'Mechanical/Typo/Insertion': 2.0,
            'Mechanical/Typo/Transposition': 2.0,
            'Mechanical/Typo/Scramble': 1.5
        }
    elif D_w > 6.0:  # Complex words
        difficulty_operators = {
            'Phonetic/Homophone': 2.5,      # Much more likely
            'Phonetic/Suffix': 2.5,
            'Orthographic/Hard Word': 4.0,  # Very likely
            'Mechanical/Typo/Deletion': 0.5, # Less likely
            'Mechanical/Typo/Insertion': 0.5,
            'Mechanical/Typo/Transposition': 0.5,
            'Mechanical/Typo/Scramble': 1.0
        }
    else:  # Medium difficulty words
        difficulty_operators = {k: 1.0 for k in T_u.keys()}  # No change

    # Apply adjustments: T_adjusted = T_u * O_d
    T_adjusted = {}
    total_adjusted = 0

    for error_type, base_prob in T_u.items():
        operator = difficulty_operators.get(error_type, 1.0)
        adjusted_prob = base_prob * operator
        T_adjusted[error_type] = adjusted_prob
        total_adjusted += adjusted_prob

    # Renormalize to ensure probabilities sum to 1
    if total_adjusted > 0:
        for error_type in T_adjusted:
            T_adjusted[error_type] /= total_adjusted

    # Convert to final format
    result = {}
    for error_type, probability in T_adjusted.items():
        result[error_type] = {
            'adjusted_probability': round(probability, 3),
            'base_probability': round(T_u[error_type], 3),
            'adjustment_factor': difficulty_operators.get(error_type, 1.0),
            'difficulty_influence': 'high' if difficulty_operators.get(error_type, 1.0) > 1.5 else 'low' if difficulty_operators.get(error_type, 1.0) < 0.8 else 'neutral'
        }

    return result


def calculate_error_proneness_index(word, user_dna):
    """Calculate Error Proneness Index (EPI) for a word - deterministic scoring."""

    # Step 1: Assess word difficulty (Dimension A: Word Characteristics)
    difficulty = assess_word_difficulty(word)
    D_w = difficulty['difficulty_score']

    # Step 2: Calculate user preference alignment (Dimension B: User DNA)
    user_error_types = user_dna['error_type_vector']

    # Extract phonetic preference (from user's historical errors)
    phonetic_preference = user_error_types.get('Phonetic/Homophone', 0) + user_error_types.get('Phonetic/Suffix', 0)
    orthographic_preference = user_error_types.get('Orthographic/Hard Word', 0)
    mechanical_preference = (user_error_types.get('Mechanical/Typo/Deletion', 0) +
                           user_error_types.get('Mechanical/Typo/Insertion', 0) +
                           user_error_types.get('Mechanical/Typo/Transposition', 0) +
                           user_error_types.get('Mechanical/Typo/Scramble', 0))

    # Step 3: Calculate word-specific features for error proneness
    word_lower = word.lower()

    # Length factor (L_w): longer words are more prone to errors
    L_w = min(len(word) / 15, 1.0)  # Normalize to 0-1, cap at 15 chars

    # Spelling complexity factor (C_w): measure orthographic difficulty
    # Count silent letters, diphthongs, consonant clusters, etc.
    silent_letters = sum(1 for char in ['b','c','d','g','h','k','p','s','t','w','x','y']
                        if char in word_lower and word_lower.count(char) > 0)  # Simplified
    diphthongs = word_lower.count('ou') + word_lower.count('oi') + word_lower.count('oo') + word_lower.count('ea')
    consonant_clusters = sum(1 for i in range(len(word_lower)-2)
                           if word_lower[i] not in 'aeiou' and word_lower[i+1] not in 'aeiou' and word_lower[i+2] not in 'aeiou')

    C_w = min((silent_letters * 0.3 + diphthongs * 0.4 + consonant_clusters * 0.3) / 3, 1.0)

    # Phonetic sensitivity factor (P_w): how prone to phonetic confusion
    # Words with similar-sounding letter combinations
    phonetic_triggers = ['tion', 'sion', 'cian', 'shun', 'sure', 'ture', 'ough', 'eigh', 'ai', 'ay']
    P_w = min(sum(1 for trigger in phonetic_triggers if trigger in word_lower) * 0.3, 1.0)

    # Step 4: Calculate Error Proneness Index (EPI)
    # EPI = (Word Difficulty × User Base Rate) × (Length Factor + Complexity Factor + Phonetic Factor) × User Preference Alignment

    E_u = user_dna['base_error_rate'] / 100.0  # Convert to decimal

    # Word difficulty contribution
    difficulty_contribution = D_w * E_u

    # Word characteristics contribution
    characteristics_contribution = (L_w * 0.4 + C_w * 0.3 + P_w * 0.3)

    # User preference alignment (weighted by error type preferences)
    preference_alignment = (phonetic_preference * P_w * 0.5 +
                          orthographic_preference * C_w * 0.3 +
                          mechanical_preference * L_w * 0.2)

    # Final EPI calculation
    EPI = difficulty_contribution * characteristics_contribution * (1 + preference_alignment)

    # Normalize to 0-1 scale for ranking
    EPI_normalized = 1 / (1 + 2.71828 ** (-EPI * 2))  # Sigmoid normalization

    return {
        'word': word,
        'error_proneness_index': round(EPI_normalized, 4),
        'raw_epi_score': round(EPI, 4),
        'dimension_breakdown': {
            'difficulty_contribution': round(difficulty_contribution, 4),
            'characteristics_contribution': round(characteristics_contribution, 4),
            'preference_alignment': round(preference_alignment, 4)
        },
        'word_features': {
            'length_factor': round(L_w, 3),
            'complexity_factor': round(C_w, 3),
            'phonetic_factor': round(P_w, 3)
        },
        'difficulty_analysis': difficulty
    }


def deterministic_error_simulation_pipeline(query, user_dna, start_time=0):
    """Complete deterministic error simulation pipeline using EPI ranking."""

    import re

    # Step 1: Tokenize query into words
    words = re.findall(r'\b\w+\b', query)

    # Step 2: Calculate EPI for each word (deterministic scoring)
    word_epi_results = []

    for word in words:
        epi_result = calculate_error_proneness_index(word, user_dna)
        word_epi_results.append(epi_result)

    # Step 3: Sort by EPI score (highest first) - deterministic ranking
    word_epi_results.sort(key=lambda x: x['error_proneness_index'], reverse=True)

    # Step 4: Always select top 1 word for error injection (user's request)
    target_error_count = 1

    # Step 5: Select top-2 words for error injection
    selected_for_error = word_epi_results[:target_error_count]

    print(f"[{time.time() - start_time:.1f}s] 🎯 Selecting top {target_error_count} EPI-ranked word for error injection", flush=True)

    # Step 6: Assign error types to selected words
    for selected in selected_for_error:
        word = selected['word']
        difficulty = selected['difficulty_analysis']
        error_types = calculate_dynamic_error_type_probabilities(user_dna, difficulty)
        selected['recommended_error_types'] = error_types

    # Step 7: Mark selection status
    for result in word_epi_results:
        result['selected_for_error'] = result in selected_for_error

    return {
        'original_query': query,
        'word_analysis': word_epi_results,
        'selected_for_error': selected_for_error,
        'total_words_analyzed': len(words),
        'error_injection_count': len(selected_for_error),
        'epi_ranking': [w['word'] for w in word_epi_results],
        'deterministic_metadata': {
            'user_base_error_rate': user_dna['base_error_rate'],
            'target_error_count': target_error_count,
            'algorithm_version': 'deterministic_epi_v1.0',
            'ranking_method': 'error_proneness_index_descending'
        }
    }


def calculate_dynamic_error_probabilities(errors, target_word=None):
    """Calculate dynamic probability distribution based on word difficulty."""
    if not errors:
        return {}

    from collections import Counter

    # Base probabilities from user's error history
    error_counts = Counter()
    total_errors = len(errors)

    for error in errors:
        category = error.get('error_category', 'Unknown')
        subcategory = error.get('error_subcategory', 'Unknown')
        key = f"{category}/{subcategory}"
        error_counts[key] += 1

    # Calculate base probabilities
    base_probabilities = {}
    for error_type, count in error_counts.items():
        base_probabilities[error_type] = count / total_errors

    # If no target word provided, return base probabilities
    if not target_word:
        return {k: {'count': error_counts[k.split('/')[0] + '/' + k.split('/')[1]],
                    'probability': round(v, 3)} for k, v in base_probabilities.items()}

    # Dynamic adjustment based on word difficulty
    difficulty = assess_word_difficulty(target_word)

    # Difficulty-based multipliers
    # For simple words (difficulty < 0.3): Reduce Phonetic/Orthographic, keep Mechanical
    # For complex words (difficulty > 0.7): Increase Phonetic/Orthographic, maintain Mechanical
    difficulty_multipliers = {}

    if difficulty < 0.3:  # Simple words
        difficulty_multipliers = {
            'Phonetic/Homophone': 0.2,      # Much less likely
            'Phonetic/Suffix': 0.2,
            'Orthographic/Hard Word': 0.1,  # Very unlikely
            'Mechanical/Typo/Deletion': 1.2, # Slightly more likely (still possible)
            'Mechanical/Typo/Insertion': 1.2,
            'Mechanical/Typo/Transposition': 1.2,
            'Mechanical/Typo/Scramble': 1.0
        }
    elif difficulty > 0.7:  # Complex words
        difficulty_multipliers = {
            'Phonetic/Homophone': 2.0,      # Much more likely
            'Phonetic/Suffix': 2.0,
            'Orthographic/Hard Word': 3.0,  # Very likely
            'Mechanical/Typo/Deletion': 0.8, # Slightly less likely (but still possible)
            'Mechanical/Typo/Insertion': 0.8,
            'Mechanical/Typo/Transposition': 0.8,
            'Mechanical/Typo/Scramble': 1.0
        }
    else:  # Medium difficulty words
        difficulty_multipliers = {k: 1.0 for k in base_probabilities.keys()}  # No change

    # Apply multipliers and renormalize
    adjusted_probabilities = {}
    total_adjusted = 0

    for error_type, base_prob in base_probabilities.items():
        multiplier = difficulty_multipliers.get(error_type, 1.0)
        adjusted_prob = base_prob * multiplier
        adjusted_probabilities[error_type] = adjusted_prob
        total_adjusted += adjusted_prob

    # Renormalize to ensure probabilities sum to 1
    if total_adjusted > 0:
        for error_type in adjusted_probabilities:
            adjusted_probabilities[error_type] /= total_adjusted

    # Convert to final format
    result = {}
    for error_type, probability in adjusted_probabilities.items():
        result[error_type] = {
            'count': error_counts[error_type],
            'probability': round(probability, 3),
            'word_difficulty': difficulty,
            'adjustment_factor': difficulty_multipliers.get(error_type, 1.0)
        }

    return result


def calculate_error_probabilities(errors):
    """Calculate probability distribution of error categories and subcategories."""
    return calculate_dynamic_error_probabilities(errors)


def extract_error_patterns(errors):
    """Extract common error patterns from the user's spelling mistakes."""
    if not errors:
        return {}

    patterns = defaultdict(list)

    for error in errors:
        original = error.get('word', '').lower()
        correct = error.get('correct', '').lower()
        error_type = error.get('error_type', 'spelling')
        reason = error.get('reason', '').lower()

        if error_type == 'spelling':
            # Categorize spelling errors
            if 'missing' in reason:
                patterns['missing_letters'].append({
                    'original': original,
                    'correct': correct,
                    'missing': reason
                })
            elif 'wrong letter' in reason or 'instead of' in reason:
                patterns['wrong_letters'].append({
                    'original': original,
                    'correct': correct,
                    'reason': reason
                })
            elif 'extra' in reason:
                patterns['extra_letters'].append({
                    'original': original,
                    'correct': correct,
                    'extra': reason
                })
            elif 'transposed' in reason:
                patterns['transposed_letters'].append({
                    'original': original,
                    'correct': correct,
                    'reason': reason
                })

    return dict(patterns)


def identify_product_category(sample_reviews):
    """Identify the main product category from sample reviews."""
    if not sample_reviews:
        return "beauty products"

    # Simple keyword-based categorization
    all_text = ' '.join(sample_reviews).lower()

    categories = {
        'skincare': ['skin care', 'moisturizer', 'cleanser', 'serum', 'toner', 'face wash'],
        'haircare': ['hair', 'shampoo', 'conditioner', 'treatment'],
        'makeup': ['makeup', 'foundation', 'lipstick', 'mascara', 'eyeshadow'],
        'bath': ['bath', 'soap', 'body wash', 'lotion'],
        'fragrance': ['perfume', 'cologne', 'fragrance', 'scent']
    }

    for category, keywords in categories.items():
        if any(keyword in all_text for keyword in keywords):
            return category

    return "beauty products"


def generate_original_query(product_category, llm_model, start_time=0):
    """Generate a clean, error-free query about the product category."""
    import sys

    print(f"[{time.time() - start_time:.1f}s] 🤖 Generating original query for category: {product_category}", flush=True)
    sys.stdout.flush()

    prompt = f"""<s> [INST] ## Task: Generate Product Search Query

**Context:**
You need to generate a realistic Amazon search query for {product_category} products. The query should be what a typical customer might search for when looking for products in this category.

**Requirements:**
- Create a natural, realistic search query that someone would actually type on Amazon
- Focus on {product_category} products specifically
- Include common product attributes or use cases
- Make it descriptive with approximately 15-25 words (not too short, not too long)
- Use proper spelling and grammar
- Make it sound like a genuine customer search intent
- Include specific product details, benefits, or use cases

**Examples for reference:**
- For skincare: "best moisturizer for dry skin with hyaluronic acid and vitamin C for anti-aging"
- For makeup: "waterproof mascara for sensitive eyes that won't cause irritation or allergies"
- For haircare: "color safe shampoo and conditioner set for damaged hair that restores moisture"

Generate a single search query for {product_category} products:
[/INST]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        query = response.content.strip()

        # Clean up the response
        if '```' in query:
            # Extract from code blocks if present
            start = query.find('```') + 3
            end = query.find('```', start)
            if end > start:
                query = query[start:end].strip()

        # Remove any extra formatting
        query = query.strip('"').strip("'").strip()

        print(f"[{time.time() - start_time:.1f}s] ✅ Generated original query: '{query}'", flush=True)
        sys.stdout.flush()

        return query

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to generate original query, using fallback: {str(e)[:50]}...", flush=True)
        sys.stdout.flush()
        # Fallback queries for different product categories
        fallback_queries = {
            'skincare': "best moisturizer for dry skin with hyaluronic acid and vitamin C for anti-aging",
            'haircare': "color safe shampoo and conditioner set for damaged hair that restores moisture",
            'makeup': "waterproof mascara for sensitive eyes that won't cause irritation or allergies"
        }
        return fallback_queries.get(product_category, f"best {product_category} products")


def analyze_query_modifications_scientific(original_query, modified_query, simulation_result, llm_model, start_time=0):
    """Analyze the modifications made using scientific simulation results."""
    import sys

    print(f"[{time.time() - start_time:.1f}s] 🔬 Analyzing modifications with scientific context...", flush=True)
    sys.stdout.flush()

    if original_query == modified_query:
        return {
            'modifications': [],
            'summary': 'No modifications were made to the query - scientific analysis determined no errors should be injected.',
            'scientific_context': simulation_result
        }

    # Format scientific simulation results for LLM
    simulation_summary = []
    simulation_summary.append(f"Words Analyzed: {simulation_result['total_words_analyzed']}")
    simulation_summary.append(f"Error Candidates: {len(simulation_result['error_candidates'])}")
    simulation_summary.append(f"Errors Injected: {len(simulation_result['selected_for_error'])}")

    scientific_context = '\n'.join(simulation_summary)

    # Detail word-by-word scientific analysis
    word_analysis_details = []
    for word_result in simulation_result['selected_for_error']:
        word = word_result['word']
        diff = word_result['difficulty_analysis']
        prob = word_result['injection_probability']
        word_analysis_details.append(f"'{word}': difficulty={diff['difficulty_score']:.2f}, injection_prob={prob:.1%}, level={diff['difficulty_level']}")

    scientific_word_details = '\n'.join(word_analysis_details) if word_analysis_details else "No words were selected for error injection"

    prompt = f"""<s> [INST] ## Task: Scientific Error Analysis

**Original Query:** "{original_query}"

**Modified Query:** "{modified_query}"

**Scientific Simulation Context:**
{scientific_context}

**Word-by-Word Scientific Analysis:**
{scientific_word_details}

**Scientific Analysis Task:**
Analyze the modifications using the mathematical coupling of User DNA + Word Difficulty:

1. **Validate scientific predictions**: Confirm that errors were applied to words selected by the algorithm
2. **Assess mathematical accuracy**: Verify that error types match the calculated difficulty-adjusted probabilities
3. **Evaluate coupling effectiveness**: Determine how well User DNA and Word Difficulty were integrated

For each modification, provide:
- **Scientific validation**: Whether the error aligns with the mathematical predictions
- **Difficulty justification**: How word difficulty score influenced the error type choice
- **User DNA alignment**: How the error matches the user's error pattern profile
- **Mathematical coupling score**: Rate the effectiveness of the User DNA + Word Difficulty integration (1-10)

**Scientific Guidelines:**
- **Reference simulation results**: Use the word-by-word analysis provided above
- **Validate probability calculations**: Ensure errors match the injection probabilities
- **Assess coupling quality**: Rate how well the scientific method predicted real user behavior

**Response Format:**
Return a JSON object with:
{{
  "modifications": [
    {{
      "original_word": "conditioner",
      "modified_word": "condishuner",
      "error_category": "Phonetic/Homophone",
      "change_description": "Replaced 'tion' with 'shun' creating a homophone-like error",
      "reason": "Following the user's highest probability error pattern (62.5% Phonetic/Homophone errors)"
    }}
  ],
  "summary": "Brief summary of all modifications and their alignment with user patterns"
}}
[/INST]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        analysis_str = response.content.strip()

        # Parse JSON response
        if '```json' in analysis_str:
            start = analysis_str.find('```json') + 7
            end = analysis_str.find('```', start)
            if end > start:
                analysis_str = analysis_str[start:end].strip()
        elif '{' in analysis_str:
            start = analysis_str.find('{')
            end = analysis_str.rfind('}') + 1
            analysis_str = analysis_str[start:end]

        try:
            analysis_result = json.loads(analysis_str)
            print(f"[{time.time() - start_time:.1f}s] ✅ Modification analysis completed", flush=True)
            return analysis_result
        except json.JSONDecodeError:
            print(f"[{time.time() - start_time:.1f}s] ⚠️ Failed to parse modification analysis JSON", flush=True)
            return {
                'modifications': [],
                'summary': 'Failed to analyze modifications due to JSON parsing error.'
            }

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to analyze modifications: {str(e)[:50]}...", flush=True)
        return {
            'modifications': [],
            'summary': f'Analysis failed: {str(e)[:100]}'
        }


def apply_deterministic_error_simulation(original_query, user_dna, llm_model, start_time=0):
    """Apply deterministically calculated error patterns using EPI ranking."""
    import sys

    # print(f"[{time.time() - start_time:.1f}s] 🎯 Applying deterministic EPI-based error simulation...", flush=True)
    sys.stdout.flush()

    # Step 1: Run deterministic error simulation pipeline
    simulation_result = deterministic_error_simulation_pipeline(original_query, user_dna, start_time)

    # EPI Analysis - no debug output in simplified mode

    # Simplified EPI analysis - no debug output needed
    sys.stdout.flush()

    # Format deterministic EPI guidance for LLM
    if simulation_result['selected_for_error']:
        error_guidance = []
        for selected in simulation_result['selected_for_error']:
            word = selected['word']
            epi_score = selected['error_proneness_index']
            error_types = selected['recommended_error_types']

            # Get top 2 most likely error types
            sorted_types = sorted(error_types.items(), key=lambda x: x[1]['adjusted_probability'], reverse=True)
            top_types = sorted_types[:2]

            type_recommendations = []
            for error_type, stats in top_types:
                prob = stats['adjusted_probability']
                influence = stats['difficulty_influence']
                type_recommendations.append(f"{error_type} ({prob:.1%}, {influence} influence)")

            error_guidance.append(f"'{word}' (EPI:{epi_score:.3f}) → {', '.join(type_recommendations)}")

        deterministic_guidance = f"""
DETERMINISTIC EPI-BASED ERROR SIMULATION:
- Error Proneness Index (EPI) ranking applied to all words
- Target error count: {simulation_result['deterministic_metadata']['target_error_count']}
- Words selected for error injection: {len(simulation_result['selected_for_error'])}
- Error injection targets: {'; '.join(error_guidance)}

**CRITICAL REQUIREMENTS:**
- Apply error to EXACTLY 1 word: the TOP 1 EPI-selected word above
- Modify the selected word individually, even if it appears in compound words
- Use the recommended error types for the word
- Do NOT modify any other words
- Do NOT skip the selected word

**Examples:**
- If "sulfate" is selected: modify it to something like "sul fate" or "sulfat"
- If "conditioner" is selected: modify it to something like "condishuner" or "conditioner"
- Always modify exactly 1 word as specified
This selection is deterministic and will always be the same for this user + query combination.
"""
    else:
        deterministic_guidance = """
DETERMINISTIC ANALYSIS: No words selected for error injection based on EPI scoring.
Return the original query unchanged.
"""

    # Format user DNA for LLM
    dna_descriptions = []
    dna_descriptions.append(f"Base Error Rate: {user_dna['base_error_rate']}%")
    for error_type, prob in user_dna['error_type_vector'].items():
        dna_descriptions.append(f"{error_type}: {prob:.1%}")

    user_dna_text = '\n'.join(dna_descriptions)

    # Create deterministic EPI prompt for LLM
    prompt = f"""<s> [INST] ## Task: Deterministic EPI-Based Error Simulation - Apply Fingerprint-Style Errors

**Original Query:** "{original_query}"

**User DNA Analysis:**
{user_dna_text}

{deterministic_guidance}

**INSTRUCTION:**
Apply spelling errors ONLY to the TOP 2 EPI-selected words using their recommended error types.
This is a DETERMINISTIC process - same user + same query will ALWAYS produce the same result.

**Error Application Rules:**
- Target ONLY the TOP 1 word specified in "Error injection targets" above (ranked by Error Proneness Index)
- Use the recommended error types for the EPI-selected word
- Maintain semantic meaning and search intent
- Apply exactly 1 error (for the selected word)

**Fingerprint Principle:**
Think of this as creating a "textual fingerprint" - this specific combination of errors
represents this user's unique writing pattern for this exact query.

**Response Format:**
Return ONLY the modified query text, nothing else. No explanations, no additional formatting.
[/INST]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        modified_query = response.content.strip()

        # Clean up response
        if '```' in modified_query:
            start = modified_query.find('```') + 3
            end = modified_query.find('```', start)
            if end > start:
                modified_query = modified_query[start:end].strip()

        modified_query = modified_query.strip('"').strip("'").strip()

        # Verify that all selected words were actually modified
        selected_words = {word_result['word'] for word_result in simulation_result['selected_for_error']}
        modified_words = set()

        # Simple check: see if any selected words appear differently in modified query
        for selected_word in selected_words:
            if selected_word not in modified_query.split():
                # Word was modified (not found in original form)
                modified_words.add(selected_word)

        if len(modified_words) < len(selected_words):
            missing_modifications = selected_words - modified_words
            # print(f"[{time.time() - start_time:.1f}s] ⚠️  Warning: Not all selected words were modified. Missing: {missing_modifications}", flush=True)
            # print(f"[{time.time() - start_time:.1f}s] 🔄 Attempting to fix incomplete modifications...", flush=True)

            # Try one more time with more explicit instructions
            fallback_prompt = f"""<s> [INST] ## Task: Fix Incomplete Error Injection

**Original Query:** "{original_query}"

**Modified Query (Incomplete):** "{modified_query}"

**CRITICAL ISSUE:** The following EPI-selected words were NOT modified: {list(missing_modifications)}

**REQUIRED FIX:** Modify EXACTLY these missing words: {list(missing_modifications)}
Apply appropriate spelling errors to each missing word while keeping all other modifications.

**IMPORTANT:** Return ONLY the corrected query text, nothing else. No explanations, no additional text.
[/INST]"""

            try:
                messages = [{"role": "user", "content": fallback_prompt}]
                response = llm_model.invoke(messages)
                corrected_query = response.content.strip().strip('"').strip("'").strip()

                if corrected_query != modified_query:
                    # print(f"[{time.time() - start_time:.1f}s] ✅ Fixed incomplete modifications", flush=True)
                    modified_query = corrected_query
                # Removed unnecessary else block
            except Exception as e:
                # print(f"[{time.time() - start_time:.1f}s] ⚠️  Failed to fix modifications: {str(e)[:50]}", flush=True)
                pass

        # print(f"[{time.time() - start_time:.1f}s] ✅ Scientific error simulation completed", flush=True)
        # print(f"[{time.time() - start_time:.1f}s] 📝 Modified query: '{modified_query}'", flush=True)
        sys.stdout.flush()

        return modified_query, simulation_result

    except Exception as e:
        # print(f"[{time.time() - start_time:.1f}s] ❌ Scientific simulation failed: {str(e)[:50]}...", flush=True)
        sys.stdout.flush()
        return original_query, simulation_result

    # Format error patterns for the LLM (examples)
    pattern_descriptions = []

    if 'missing_letters' in error_patterns and error_patterns['missing_letters']:
        examples = error_patterns['missing_letters'][:3]  # Limit to 3 examples
        example_strs = [f"{ex['original']}→{ex['correct']}" for ex in examples]
        pattern_descriptions.append(f"Missing letters: {', '.join(example_strs)}")

    if 'wrong_letters' in error_patterns and error_patterns['wrong_letters']:
        examples = error_patterns['wrong_letters'][:3]
        example_strs = [f"{ex['original']}→{ex['correct']}" for ex in examples]
        pattern_descriptions.append(f"Wrong letters: {', '.join(example_strs)}")

    if 'extra_letters' in error_patterns and error_patterns['extra_letters']:
        examples = error_patterns['extra_letters'][:3]
        example_strs = [f"{ex['original']}→{ex['correct']}" for ex in examples]
        pattern_descriptions.append(f"Extra letters: {', '.join(example_strs)}")

    if 'transposed_letters' in error_patterns and error_patterns['transposed_letters']:
        examples = error_patterns['transposed_letters'][:3]
        example_strs = [f"{ex['original']}→{ex['correct']}" for ex in examples]
        pattern_descriptions.append(f"Transposed letters: {', '.join(example_strs)}")

    patterns_text = '; '.join(pattern_descriptions) if pattern_descriptions else "No specific patterns available"

    prompt = f"""<s> [INST] ## Task: Apply User Error Patterns to Query

**Original Query:** "{original_query}"

**User Error Probability Distribution:**
{probabilities_text}

**User Error Pattern Examples:**
{patterns_text}

**Task:**
Create a modified version of the original query that incorporates this user's typical spelling error patterns, weighted by their probability distribution. The modified query should:

1. **Maintain the same semantic meaning** - it should search for the same type of products
2. **Include realistic spelling errors** based on the user's error probability distribution above
3. **Be something the user might actually type** with their error habits
4. **Keep the overall structure and intent** of the original query

**DYNAMIC ERROR APPLICATION BASED ON WORD DIFFICULTY:**

**Word Difficulty Guidelines:**
- **Simple words** (short, common, familiar): Prioritize Mechanical/Typo errors, minimize Phonetic/Orthographic errors
  - Examples: "hair", "good", "free", "best" → Focus on typos like "wieghing" instead of "weighing"
- **Complex words** (long, technical, unfamiliar): Prioritize Phonetic and Orthographic errors
  - Examples: "conditioner", "sulfate", "hyaluronic" → Apply phonetic spelling like "condishuner" or orthographic errors

**Probability Distribution (Adjusted by Word Difficulty):**
{probabilities_text}

**CRITICAL REQUIREMENTS:**
- **Apply 1-2 targeted errors** based on word difficulty and adjusted probabilities
- **Match error types to word complexity**: Simple words get typos, complex words get phonetic/orthographic errors
- **Follow the dynamic probability distribution** shown above with difficulty adjustments
- **Be realistic**: Users rarely misspell simple words phonetically, but often struggle with complex terminology

**Example:**
Original: "best moisturizer for dry skin with hyaluronic acid"
Modified: "best moiturizer for dry skin with hyularonic acid" (applying Phonetic/Homophone pattern)

Return only the modified query with deliberate spelling errors, nothing else.
[/INST]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        modified_query = response.content.strip()

        # Clean up the response
        if '```' in modified_query:
            start = modified_query.find('```') + 3
            end = modified_query.find('```', start)
            if end > start:
                modified_query = modified_query[start:end].strip()

        modified_query = modified_query.strip('"').strip("'").strip()

        print(f"[{time.time() - start_time:.1f}s] ✅ Generated modified query: '{modified_query}'", flush=True)
        sys.stdout.flush()

        return modified_query

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to generate modified query, using fallback: {str(e)[:50]}...", flush=True)
        sys.stdout.flush()
        return original_query  # Fallback to original


def generate_error_aware_queries(input_file, output_file, llm_model, start_time=0):
    """Legacy function for single query processing - now redirects to batch processing."""
    import sys

    print(f"[{time.time() - start_time:.1f}s] 📖 Reading analysis file: {input_file}", flush=True)
    sys.stdout.flush()

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"[{time.time() - start_time:.1f}s] ❌ Input analysis file does not exist: {input_file}", flush=True)
        sys.stdout.flush()
        return None

    # Read analysis data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        print(f"[{time.time() - start_time:.1f}s] ✅ Loaded analysis for user: {analysis_data.get('user_id', 'unknown')}", flush=True)
        print(f"[{time.time() - start_time:.1f}s] 📊 Found {analysis_data.get('error_count', 0)} spelling errors", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to read analysis file: {e}", flush=True)
        sys.stdout.flush()
        return None

    # Extract error patterns and calculate probabilities
    errors = analysis_data.get('spelling_errors', [])
    error_patterns = extract_error_patterns(errors)

    # Scientific User DNA assessment
    user_dna = assess_user_dna(errors)
    print(f"[{time.time() - start_time:.1f}s] 🧬 User DNA Analysis:", flush=True)
    print(f"[{time.time() - start_time:.1f}s]   - Base Error Rate: {user_dna['base_error_rate']}%", flush=True)
    print(f"[{time.time() - start_time:.1f}s]   - Total Errors: {user_dna['total_errors']}", flush=True)
    for error_type, prob in user_dna['error_type_vector'].items():
        print(f"[{time.time() - start_time:.1f}s]   - {error_type}: {prob:.3f}", flush=True)
    sys.stdout.flush()

    error_probabilities = calculate_error_probabilities(errors)

    print(f"[{time.time() - start_time:.1f}s] 🔍 Identified {len(error_patterns)} error pattern categories", flush=True)
    for pattern_type, pattern_list in error_patterns.items():
        print(f"[{time.time() - start_time:.1f}s]   - {pattern_type}: {len(pattern_list)} examples", flush=True)
    sys.stdout.flush()

    print(f"[{time.time() - start_time:.1f}s] 📊 Error probability distribution:", flush=True)
    for error_type, stats in error_probabilities.items():
        print(f"[{time.time() - start_time:.1f}s]   - {error_type}: {stats['probability']:.3f} ({stats['count']} instances)", flush=True)
    sys.stdout.flush()

    # Identify product category
    sample_reviews = analysis_data.get('sample_reviews', [])
    product_category = identify_product_category(sample_reviews)

    print(f"[{time.time() - start_time:.1f}s] 🏷️ Identified product category: {product_category}", flush=True)
    sys.stdout.flush()

    # Generate original query
    original_query = generate_original_query(product_category, llm_model, start_time)

    # Generate error-modified query using deterministic EPI simulation
    modified_query, simulation_result = apply_deterministic_error_simulation(original_query, user_dna, llm_model, start_time)

    # Analyze the modifications made with deterministic EPI context
    print(f"[{time.time() - start_time:.1f}s] 🔍 Debug - Original: '{original_query[:50]}...'", flush=True)
    print(f"[{time.time() - start_time:.1f}s] 🔍 Debug - Modified: '{modified_query[:50]}...'", flush=True)
    print(f"[{time.time() - start_time:.1f}s] 🔍 Debug - Are they equal? {original_query == modified_query}", flush=True)
    sys.stdout.flush()

    modification_analysis = analyze_deterministic_modifications(original_query, modified_query, simulation_result, llm_model, start_time)

    # Print modification analysis details
    print(f"[{time.time() - start_time:.1f}s] 📝 Modification Analysis Results:", flush=True)
    sys.stdout.flush()

    if modification_analysis.get('modifications'):
        for i, mod in enumerate(modification_analysis['modifications'], 1):
            print(f"[{time.time() - start_time:.1f}s]   {i}. '{mod['original_word']}' → '{mod['modified_word']}' ({mod['error_category']})", flush=True)
            explanation = mod.get('explanation', mod.get('reason', 'No explanation provided'))
            # Print first 150 characters of explanation to avoid log overflow
            explanation_preview = explanation[:150] + ('...' if len(explanation) > 150 else '')
            print(f"[{time.time() - start_time:.1f}s]      Explanation: {explanation_preview}", flush=True)
            print(f"[{time.time() - start_time:.1f}s]", flush=True)  # Empty line for readability
            sys.stdout.flush()
    else:
        print(f"[{time.time() - start_time:.1f}s]   No modifications detected", flush=True)
        sys.stdout.flush()

    if 'summary' in modification_analysis:
        print(f"[{time.time() - start_time:.1f}s]   📋 Summary: {modification_analysis['summary']}", flush=True)
        sys.stdout.flush()

    # Prepare result with scientific coupling analysis
    result = {
        'user_id': analysis_data.get('user_id'),
        'product_category': product_category,
        'error_patterns': error_patterns,
        'error_probabilities': error_probabilities,
        'error_count': analysis_data.get('error_count', 0),
        'user_dna': user_dna,
        'deterministic_simulation': simulation_result,
        'epi_methodology': {
            'formula': 'EPI = (D_w × E_u) × (L_w + C_w + P_w) × (1 + User_Preference_Alignment)',
            'variables': {
                'D_w': 'Word difficulty score (length × frequency × phonetic_complexity ÷ 100)',
                'E_u': 'User base error rate (percentage converted to decimal)',
                'L_w': 'Length factor (normalized word length 0-1)',
                'C_w': 'Spelling complexity factor (silent letters, diphthongs, consonant clusters)',
                'P_w': 'Phonetic sensitivity factor (pronunciation triggers)',
                'User_Preference_Alignment': 'Weighted alignment with user error type preferences'
            },
            'deterministic_principle': 'User_DNA ⊗ Word_Characteristics → Error_Proneness_Index → Top-N_Selection',
            'epi_ranges': {
                'low_proneness': '< 0.3 (words user rarely misspells)',
                'medium_proneness': '0.3-0.7 (moderate error likelihood)',
                'high_proneness': '> 0.7 (words user almost always misspells)'
            },
            'selection_mechanism': 'Deterministic ranking by EPI score, select top-N words where N = f(user_error_rate × query_length)'
        },
        'queries': {
            'original_query': original_query,
            'error_modified_query': modified_query,
            'semantic_preservation': True,
            'error_intent': 'Scientific simulation using mathematical coupling of User DNA + Word Difficulty'
        },
        'modification_analysis': modification_analysis,
        'metadata': {
            'generated_at': time.time(),
            'analysis_source': input_file,
            'llm_model': 'SiliconFlow',
            'deterministic_epi': True,
            'algorithm_version': 'deterministic_epi_v1.0',
            'fingerprint_principle': 'Same user + same query = same errors (deterministic)'
        }
    }

    # Save results
    try:
        print(f"[{time.time() - start_time:.1f}s] 💾 Saving query results...", flush=True)
        sys.stdout.flush()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{time.time() - start_time:.1f}s] ✅ Results saved to: {output_file}", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to save results: {e}", flush=True)
        sys.stdout.flush()
        return None

    print(f"[{time.time() - start_time:.1f}s] 🎉 Query generation completed!", flush=True)
    print(f"[{time.time() - start_time:.1f}s] 📋 Original: '{original_query}'", flush=True)
    print(f"[{time.time() - start_time:.1f}s] 🔄 Modified: '{modified_query}'", flush=True)
    sys.stdout.flush()

    return result


def process_batch_queries(input_csv_file, output_json_file, llm_model, start_time=0):
    """Process all queries from CSV file and generate error-aware versions."""
    import sys
    import pandas as pd

    print(f"📖 [{time.time() - start_time:.1f}s] Reading queries from CSV: {input_csv_file}", flush=True)
    sys.stdout.flush()

    # Read CSV file
    try:
        df = pd.read_csv(input_csv_file)
        queries = df['query'].tolist()
        query_ids = df['id'].tolist() if 'id' in df.columns else list(range(len(queries)))
        print(f"✅ [{time.time() - start_time:.1f}s] Loaded {len(queries)} queries from CSV", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to read CSV file: {e}", flush=True)
        sys.stdout.flush()
        return None

    # Store df for later use in CSV generation
    global csv_df
    csv_df = df

    # Process queries concurrently with simplified output
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = []
    successful_conversions = 0
    lock = threading.Lock()  # For thread-safe updates

    # Limit concurrent requests to avoid overwhelming the API
    max_concurrent = min(100, len(queries))
    print(f"🔄 [{time.time() - start_time:.1f}s] Processing {len(queries)} queries with max {max_concurrent} concurrent requests", flush=True)
    sys.stdout.flush()

    def process_single_query(query_data):
        """Process a single query (thread-safe function)."""
        query_id, query = query_data
        thread_start_time = time.time()

        try:
            # Generate single error-aware query using our EPI system
            result = generate_single_error_aware_query(query, query_id, llm_model, thread_start_time)

            if result:
                return result, None  # Success
            else:
                return None, f"Failed to process query {query_id}"  # Failure

        except Exception as e:
            error_msg = f"Error processing query {query_id}: {str(e)[:100]}..."
            return None, error_msg

    # Submit all queries for concurrent processing
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Create future mappings
        future_to_query = {
            executor.submit(process_single_query, (query_id, query)): (query_id, query)
            for query_id, query in zip(query_ids, queries)
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_query):
            query_id, query = future_to_query[future]

            try:
                result, error_msg = future.result()

                if result:
                    results.append(result)
                    with lock:
                        successful_conversions += 1

                    # Simplified output format - always show this format for every query
                    print(f"\n{'='*80}", flush=True)
                    print(f"Query ID: {query_id}", flush=True)
                    print(f"Original: {query}", flush=True)

                    # Always show selected words if available
                    selected_words = result.get('simulation_result', {}).get('selected_for_error', [])
                    if selected_words:
                        selected_word_names = [w['word'] for w in selected_words]
                        print(f"Selected words: {', '.join(selected_word_names)}", flush=True)
                    else:
                        print(f"Selected words: None", flush=True)

                    # Show modifications if available
                    modifications = result.get('modification_analysis', {}).get('modifications', [])
                    if modifications:
                        print(f"Modifications:", flush=True)
                        for j, mod in enumerate(modifications, 1):
                            print(f"  {j}. '{mod['original_word']}' → '{mod['modified_word']}' ({mod['error_category']})", flush=True)
                            explanation = mod.get('explanation', 'No explanation provided')
                            # Show complete explanation
                            print(f"     Reason: {explanation}", flush=True)

                        print(f"Modified: {result['error_modified_query']}", flush=True)
                    else:
                        print(f"Modifications: None", flush=True)
                        print(f"Modified: {result.get('error_modified_query', query)}", flush=True)

                else:
                    print(f"❌ Query {query_id}: {error_msg}", flush=True)
                    # Add original query as fallback
                    results.append({
                        'query_id': query_id,
                        'original_query': query,
                        'error_modified_query': query,  # fallback to original
                        'modification_status': 'failed',
                        'error_message': error_msg
                    })

            except Exception as e:
                print(f"❌ [{time.time() - start_time:.1f}s] Query {query_id}: Exception - {str(e)[:100]}...", flush=True)
                results.append({
                    'query_id': query_id,
                    'original_query': query,
                    'error_modified_query': query,  # fallback to original
                    'modification_status': 'failed',
                    'error_message': str(e)
                })

        # Final progress update
        print(f"📊 [{time.time() - start_time:.1f}s] All queries submitted. Waiting for completion...", flush=True)

        try:
            # Generate single error-aware query using our EPI system
            result = generate_single_error_aware_query(query, query_id, llm_model, start_time)

            if result:
                results.append(result)
                successful_conversions += 1

                # Simplified output format
                print(f"\n{'='*80}", flush=True)
                print(f"Query ID: {query_id}", flush=True)
                print(f"Original: {query}", flush=True)

                if result.get('modification_analysis', {}).get('modifications'):
                    # Show selected words for modification
                    selected_words = result.get('simulation_result', {}).get('selected_for_error', [])
                    if selected_words:
                        selected_word_names = [w['word'] for w in selected_words]
                        print(f"Selected words: {', '.join(selected_word_names)}", flush=True)

                    # Show modifications and explanations
                    print(f"Modifications:", flush=True)
                    for j, mod in enumerate(result['modification_analysis']['modifications'], 1):
                        print(f"  {j}. '{mod['original_word']}' → '{mod['modified_word']}' ({mod['error_category']})", flush=True)
                        explanation = mod.get('explanation', 'No explanation provided')
                        # Show complete explanation
                        print(f"     Reason: {explanation}", flush=True)

                    print(f"Modified: {result['error_modified_query']}", flush=True)
                else:
                    print(f"Modifications: None", flush=True)
                    print(f"Modified: {result.get('error_modified_query', query)}", flush=True)

            else:
                print(f"⚠️ [{time.time() - start_time:.1f}s] Failed to process query {query_id}", flush=True)
                # Add original query as fallback
                results.append({
                    'query_id': query_id,
                    'original_query': query,
                    'error_modified_query': query,  # fallback to original
                    'modification_status': 'failed',
                    'error_message': 'Processing failed'
                })

        except Exception as e:
            print(f"❌ [{time.time() - start_time:.1f}s] Query {query_id}: Exception - {str(e)[:100]}...", flush=True)
            results.append({
                'query_id': query_id,
                'original_query': query,
                'error_modified_query': query,  # fallback to original
                'modification_status': 'failed',
                'error_message': str(e)
            })

        # Final progress update
        print(f"📊 [{time.time() - start_time:.1f}s] All queries submitted. Waiting for completion...", flush=True)

    try:
        import json
        output_data = {
            'metadata': {
                'total_queries_processed': len(queries),
                'successful_conversions': successful_conversions,
                'processing_timestamp': time.time(),
                'algorithm_version': 'deterministic_epi_batch_v1.0'
            },
            'results': results
        }

        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 [{time.time() - start_time:.1f}s] JSON results saved to: {output_json_file}", flush=True)

        # Generate CSV output with specific name and location
        csv_output_file = "/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/error_aware_variants_81.csv"

        # Create CSV data with same format as input CSV
        csv_results = []
        for i, (query_id, original_query) in enumerate(zip(query_ids, queries)):
            # Find corresponding result for this query
            result = next((r for r in results if r['query_id'] == query_id), None)

            if result:
                modified_query = result.get('error_modified_query', original_query)
            else:
                modified_query = original_query

            # Get original row data
            original_row = csv_df.iloc[i]
            csv_row = {
                'id': query_id,
                'query': modified_query,  # Use modified query instead of original
                'answer_ids': original_row['answer_ids'],
                'answer_ids_source': original_row['answer_ids_source']
            }
            csv_results.append(csv_row)

        # Save CSV results with same format as input
        import csv
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'query', 'answer_ids', 'answer_ids_source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)

        print(f"💾 [{time.time() - start_time:.1f}s] CSV results saved to: {csv_output_file}", flush=True)
        print(f"📊 [{time.time() - start_time:.1f}s] Final Summary: {successful_conversions}/{len(queries)} queries successfully processed", flush=True)
        sys.stdout.flush()

        return output_data

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to save batch results: {e}", flush=True)
        sys.stdout.flush()
        return None


def generate_single_error_aware_query(query, query_id, llm_model, start_time=0):
    """Generate error-aware version for a single query."""
    try:
        # Load user DNA (using fixed analysis file for all queries)
        analysis_data = load_analysis_data("/home/wlia0047/ar57_scratch/wenyu/style_analysis_first_user.json")

        if not analysis_data:
            return None

        # Extract error patterns and calculate probabilities
        errors = analysis_data.get('spelling_errors', [])
        error_patterns = extract_error_patterns(errors)

        # Scientific User DNA assessment
        user_dna = assess_user_dna(errors)

        error_probabilities = calculate_error_probabilities(errors)

        # Generate error-modified query using deterministic EPI simulation
        modified_query, simulation_result = apply_deterministic_error_simulation(query, user_dna, llm_model, start_time)

        # Analyze the modifications
        modification_analysis = analyze_deterministic_modifications(query, modified_query, simulation_result, llm_model, start_time)

        # Prepare result for this query
        result = {
            'query_id': query_id,
            'original_query': query,
            'error_modified_query': modified_query,
            'modification_status': 'success',
            'user_dna': user_dna,
            'simulation_result': simulation_result,
            'modification_analysis': modification_analysis
        }

        return result

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Error in single query processing: {str(e)[:100]}...", flush=True)
        return None


def load_analysis_data(analysis_file):
    """Load and cache analysis data."""
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load analysis file: {e}")
        return None


def process_synthesized_queries(input_csv_file, output_json_file, llm_model, start_time=0):
    """Process all queries from synthesized CSV file and generate error-aware versions."""
    import sys
    import pandas as pd
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"📖 [{time.time() - start_time:.1f}s] Reading SYNTHESIZED queries from CSV: {input_csv_file}", flush=True)
    sys.stdout.flush()

    # Read CSV file
    try:
        df = pd.read_csv(input_csv_file)
        queries = df['query'].tolist()
        query_ids = df['id'].tolist() if 'id' in df.columns else list(range(len(queries)))
        print(f"✅ [{time.time() - start_time:.1f}s] Loaded {len(queries)} SYNTHESIZED queries from CSV", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to read SYNTHESIZED CSV file: {e}", flush=True)
        sys.stdout.flush()
        return None

    # Store df for later use in CSV generation
    global csv_df
    csv_df = df

    # Process queries concurrently with simplified output
    results = []
    successful_conversions = 0
    lock = threading.Lock()  # For thread-safe updates

    # Limit concurrent requests to avoid overwhelming the API
    max_concurrent = min(100, len(queries))
    print(f"🔄 [{time.time() - start_time:.1f}s] Processing {len(queries)} SYNTHESIZED queries with max {max_concurrent} concurrent requests", flush=True)
    sys.stdout.flush()

    def process_single_synthesized_query(query_data):
        """Process a single synthesized query (thread-safe function)."""
        query_id, query = query_data
        thread_start_time = time.time()

        try:
            # Generate single error-aware query using our EPI system
            result = generate_single_error_aware_query(query, query_id, llm_model, thread_start_time)

            if result:
                return result, None  # Success
            else:
                return None, f"Failed to process synthesized query {query_id}"  # Failure

        except Exception as e:
            error_msg = f"Error processing synthesized query {query_id}: {str(e)[:100]}..."
            return None, error_msg

    # Submit all queries for concurrent processing
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Create future mappings
        future_to_query = {
            executor.submit(process_single_synthesized_query, (query_id, query)): (query_id, query)
            for query_id, query in zip(query_ids, queries)
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_query):
            query_id, query = future_to_query[future]

            try:
                result, error_msg = future.result()

                if result:
                    results.append(result)
                    with lock:
                        successful_conversions += 1

                    # Simplified output format - always show this format for every query
                    print(f"\n{'='*80}", flush=True)
                    print(f"SYNTHESIZED Query ID: {query_id}", flush=True)
                    print(f"Original: {query}", flush=True)

                    # Always show selected words if available
                    selected_words = result.get('simulation_result', {}).get('selected_for_error', [])
                    if selected_words:
                        selected_word_names = [w['word'] for w in selected_words]
                        print(f"Selected words: {', '.join(selected_word_names)}", flush=True)
                    else:
                        print(f"Selected words: None", flush=True)

                    # Show modifications if available
                    modifications = result.get('modification_analysis', {}).get('modifications', [])
                    if modifications:
                        print(f"Modifications:", flush=True)
                        for j, mod in enumerate(modifications, 1):
                            print(f"  {j}. '{mod['original_word']}' → '{mod['modified_word']}' ({mod['error_category']})", flush=True)
                            explanation = mod.get('explanation', 'No explanation provided')
                            # Show complete explanation
                            print(f"     Reason: {explanation}", flush=True)

                        print(f"Modified: {result['error_modified_query']}", flush=True)
                    else:
                        print(f"Modifications: None", flush=True)
                        print(f"Modified: {result.get('error_modified_query', query)}", flush=True)

                else:
                    print(f"❌ SYNTHESIZED Query {query_id}: {error_msg}", flush=True)
                    # Add original query as fallback
                    results.append({
                        'query_id': query_id,
                        'original_query': query,
                        'error_modified_query': query,  # fallback to original
                        'modification_status': 'failed',
                        'error_message': error_msg
                    })

            except Exception as e:
                print(f"❌ [{time.time() - start_time:.1f}s] SYNTHESIZED Query {query_id}: Exception - {str(e)[:100]}...", flush=True)
                results.append({
                    'query_id': query_id,
                    'original_query': query,
                    'error_modified_query': query,  # fallback to original
                    'modification_status': 'failed',
                    'error_message': str(e)
                })

    # Final progress update
    print(f"📊 [{time.time() - start_time:.1f}s] All SYNTHESIZED queries submitted. Waiting for completion...", flush=True)

    try:
        import json
        output_data = {
            'metadata': {
                'total_queries_processed': len(queries),
                'successful_conversions': successful_conversions,
                'processing_timestamp': time.time(),
                'algorithm_version': 'deterministic_epi_synthesized_batch_v1.0',
                'data_source': 'synthesized'
            },
            'results': results
        }

        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 [{time.time() - start_time:.1f}s] SYNTHESIZED JSON results saved to: {output_json_file}", flush=True)

        # Generate CSV output with specific name and location
        csv_output_file = "/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/error_aware_synthesized_variants_100.csv"

        # Create CSV data with same format as input CSV
        csv_results = []
        for i, (query_id, original_query) in enumerate(zip(query_ids, queries)):
            # Find corresponding result for this query
            result = next((r for r in results if r['query_id'] == query_id), None)

            if result:
                modified_query = result.get('error_modified_query', original_query)
            else:
                modified_query = original_query

            # Get original row data
            original_row = csv_df.iloc[i]
            csv_row = {
                'id': query_id,
                'query': modified_query,  # Use modified query instead of original
                'answer_ids': original_row['answer_ids']
                # Note: synthesized data doesn't have answer_ids_source column
            }
            csv_results.append(csv_row)

        # Save CSV results with format matching synthesized data (no answer_ids_source)
        import csv
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'query', 'answer_ids']  # No answer_ids_source for synthesized
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)

        print(f"💾 [{time.time() - start_time:.1f}s] SYNTHESIZED CSV results saved to: {csv_output_file}", flush=True)
        print(f"📊 [{time.time() - start_time:.1f}s] SYNTHESIZED Final Summary: {successful_conversions}/{len(queries)} queries successfully processed", flush=True)
        sys.stdout.flush()

        return output_data

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to save SYNTHESIZED batch results: {e}", flush=True)
        sys.stdout.flush()
        return None


def main():
    """Main function - now supports batch processing of both human generated and synthesized queries."""
    import sys
    import time

    start_time = time.time()
    print(f"🚀 [{time.time() - start_time:.1f}s] Starting Batch Error-Aware Query Generation for BOTH datasets...", flush=True)
    sys.stdout.flush()

    print(f"🤖 [{time.time() - start_time:.1f}s] Initializing SiliconFlow model...", flush=True)
    sys.stdout.flush()

    try:
        llm_model = get_gm_model()
        print(f"✅ [{time.time() - start_time:.1f}s] SiliconFlow model initialized successfully", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to initialize model: {e}", flush=True)
        print(f"💥 LLM initialization failed. This script requires LLM to function. Exiting.", flush=True)
        sys.stdout.flush()
        sys.exit(1)

    # Process human generated queries
    print(f"\n{'='*100}", flush=True)
    print(f"🎯 [{time.time() - start_time:.1f}s] Processing HUMAN GENERATED queries...", flush=True)
    print(f"{'='*100}", flush=True)
    sys.stdout.flush()

    input_csv_human = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_human_generated_eval.csv"
    output_json_human = "/home/wlia0047/ar57_scratch/wenyu/error_aware_queries_batch.json"

    print(f"📁 [{time.time() - start_time:.1f}s] Human Input CSV: {input_csv_human}", flush=True)
    print(f"💾 [{time.time() - start_time:.1f}s] Human Output JSON: {output_json_human}", flush=True)
    sys.stdout.flush()

    result_human = process_batch_queries(input_csv_human, output_json_human, llm_model, start_time)

    # Process synthesized queries
    print(f"\n{'='*100}", flush=True)
    print(f"🎯 [{time.time() - start_time:.1f}s] Processing SYNTHESIZED queries...", flush=True)
    print(f"{'='*100}", flush=True)
    sys.stdout.flush()

    input_csv_synth = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_synthesized_100.csv"
    output_json_synth = "/home/wlia0047/ar57_scratch/wenyu/error_aware_synthesized_queries_batch.json"

    print(f"📁 [{time.time() - start_time:.1f}s] Synthesized Input CSV: {input_csv_synth}", flush=True)
    print(f"💾 [{time.time() - start_time:.1f}s] Synthesized Output JSON: {output_json_synth}", flush=True)
    sys.stdout.flush()

    result_synth = process_synthesized_queries(input_csv_synth, output_json_synth, llm_model, start_time)

    print(f"\n{'='*100}", flush=True)
    print(f"🏁 [{time.time() - start_time:.1f}s] ALL processing completed!", flush=True)
    print(f"📊 [{time.time() - start_time:.1f}s] Human generated: {result_human['metadata']['successful_conversions'] if result_human else 0}/83 queries", flush=True)
    print(f"📊 [{time.time() - start_time:.1f}s] Synthesized: {result_synth['metadata']['successful_conversions'] if result_synth else 0}/102 queries", flush=True)
    print(f"{'='*100}", flush=True)
    sys.stdout.flush()

    return {
        'human_generated': result_human,
        'synthesized': result_synth
    }


def analyze_deterministic_modifications(original_query, modified_query, simulation_result, llm_model, start_time=0):
    """Analyze modifications using deterministic EPI context."""
    import sys

    # print(f"[{time.time() - start_time:.1f}s] 🎯 Analyzing deterministic EPI-based modifications...", flush=True)
    sys.stdout.flush()

    if original_query == modified_query:
        return {
            'modifications': [],
            'summary': 'No modifications were made - EPI analysis determined no errors should be injected for this user-query combination.',
            'deterministic_context': simulation_result
        }

    # Format deterministic EPI results for LLM
    epi_summary = []
    epi_summary.append(f"EPI Ranking Applied: {' → '.join(simulation_result['epi_ranking'][:5])}")
    epi_summary.append(f"Target Error Count: {simulation_result['deterministic_metadata']['target_error_count']}")
    epi_summary.append(f"Errors Injected: {len(simulation_result['selected_for_error'])}")

    deterministic_context = '\n'.join(epi_summary)

    # Detail EPI scores for selected words
    epi_details = []
    for selected in simulation_result['selected_for_error']:
        word = selected['word']
        epi_score = selected['error_proneness_index']
        breakdown = selected['dimension_breakdown']
        epi_details.append(f"'{word}': EPI={epi_score:.3f} (难度×偏好×特征={breakdown['difficulty_contribution']:.3f}×{breakdown['preference_alignment']:.3f}×{breakdown['characteristics_contribution']:.3f})")

    epi_word_details = '\n'.join(epi_details) if epi_details else "No words were selected for error injection"

    prompt = f"""<s> [INST] ## Task: Deterministic EPI Modification Analysis

**Original Query:** "{original_query}"

**Modified Query:** "{modified_query}"

**Deterministic EPI Context:**
{deterministic_context}

**EPI-Selected Words Analysis:**
{epi_word_details}

**Analysis Task:**
Provide DETAILED explanations for each modification made:

1. **Per-Modification Analysis**: For each modified word, explain:
   - **What was changed**: Original word → Modified word
   - **Why this word was chosen**: EPI ranking and selection criteria
   - **Error type reasoning**: Which error category it represents and why
   - **User behavior alignment**: How this matches the user's historical error patterns
   - **Cognitive justification**: Why this specific error makes sense for this user and word

2. **Overall Assessment**:
   - **EPI Prediction Accuracy**: How well the modifications match EPI selections
   - **User DNA Alignment**: Error type distribution match with user history
   - **Deterministic Consistency**: Reliability of this error pattern for this user
   - **Fingerprint Quality**: How well these errors represent user's unique writing style

**Format each explanation clearly with specific reasoning and evidence from the EPI analysis.**

**REQUIRED JSON FORMAT:**
Return a JSON object with this exact structure:
{{
  "modifications": [
    {{
      "original_word": "sulfate",
      "modified_word": "sul fate",
      "error_category": "Orthographic/Hard Word",
      "explanation": "Detailed explanation of why this modification was made, including EPI ranking, user DNA alignment, and cognitive justification"
    }},
  ],
  "overall_assessment": "Summary of the analysis with key insights about user behavior patterns"
}}

**CRITICAL: Include EXACTLY 1 modification in the array, for the EPI-selected word.**

**Response Format:**
Return a JSON object with modifications analysis and fingerprint assessment.

**IMPORTANT:** Do NOT include any additional text or explanations outside the JSON structure.
[/INST]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        analysis_str = response.content.strip()

        # Parse JSON response
        if '```json' in analysis_str:
            start = analysis_str.find('```json') + 7
            end = analysis_str.find('```', start)
            if end > start:
                analysis_str = analysis_str[start:end].strip()
        elif '{' in analysis_str:
            start = analysis_str.find('{')
            end = analysis_str.rfind('}') + 1
            analysis_str = analysis_str[start:end]

        try:
            analysis_result = json.loads(analysis_str)
            # print(f"[{time.time() - start_time:.1f}s] ✅ Deterministic modification analysis completed", flush=True)
            # print(f"[{time.time() - start_time:.1f}s] 📊 Analysis result keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}", flush=True)
            return analysis_result
        except json.JSONDecodeError as e:
            # print(f"[{time.time() - start_time:.1f}s] ⚠️ Failed to parse deterministic analysis JSON: {str(e)}", flush=True)
            # print(f"[{time.time() - start_time:.1f}s] 📄 Raw LLM response (first 200 chars): {analysis_str[:200]}", flush=True)
            # Try to extract summary if JSON failed
            if 'modifications' in locals() and locals()['modifications']:
                return {
                    'modifications': locals()['modifications'],
                    'summary': f'JSON parsing failed, but found {len(locals()["modifications"])} modifications',
                    'deterministic_context': simulation_result
                }
            else:
                return {
                    'modifications': [],
                    'summary': f'JSON parsing failed. Raw response: {analysis_str[:100]}...',
                    'deterministic_context': simulation_result
                }

    except Exception as e:
        # print(f"[{time.time() - start_time:.1f}s] ❌ Deterministic analysis failed: {str(e)[:50]}...", flush=True)
        return {
            'modifications': [],
            'summary': f'Deterministic analysis failed: {str(e)[:100]}',
            'deterministic_context': simulation_result
        }


if __name__ == "__main__":
    main()
