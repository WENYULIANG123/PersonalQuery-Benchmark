#!/usr/bin/env python3
"""
Amazon Review Spelling Error Analysis using SiliconFlow LLM

This script analyzes spelling errors in Amazon beauty product reviews by:
1. Extracting and classifying spelling errors using mandatory error categorization system
2. Validating error classifications and explanations
3. Generating error-aware query modifications for user behavior simulation
"""

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure stark/code is on Python path so we can import model.py
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
from model import get_gm_model, call_llm_with_retry

import nltk
from nltk.tokenize import sent_tokenize

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


# Simple configuration
INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json"
OUTPUT_FILE = "/home/wlia0047/ar57_scratch/wenyu/spelling_error_style_analysis.json"


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


def classify_error_from_reason(error):
    """Classify error into exactly ONE category and subcategory based on the reason provided."""
    original = error['original_word']
    correction = error['corrected_word']
    reason = error.get('reason', '').lower()

    # Skip if this is not a real error
    if original == correction:
        return None

    # Analyze reason to determine exactly ONE category and subcategory
    error_category = None
    error_subcategory = None

    # Priority order: check most specific patterns first
    # Mechanical/Typo patterns
    if any(keyword in reason for keyword in ['missing letter', 'omitted letter', 'letter omitted', 'forgot to type', 'incomplete key press']):
        error_category = "Mechanical/Typo"
        error_subcategory = "Deletion"
    elif any(keyword in reason for keyword in ['extra letter', 'additional letter', 'inserted letter', 'added letter', 'key sticking']):
        error_category = "Mechanical/Typo"
        error_subcategory = "Insertion"
    elif any(keyword in reason for keyword in ['letters swapped', 'transposed letters', 'wrong order', 'adjacent letters']):
        error_category = "Mechanical/Typo"
        error_subcategory = "Transposition"
    elif any(keyword in reason for keyword in ['multiple errors', 'combination of errors', 'chaotic', 'scrambled']):
        error_category = "Mechanical/Typo"
        error_subcategory = "Scramble"

    # Phonetic patterns (only if not already classified as Mechanical)
    elif any(keyword in reason for keyword in ['sounds like', 'homophone', 'same pronunciation', 'sounds the same', 'sound alike']):
        error_category = "Phonetic"
        error_subcategory = "Homophone"
    elif any(keyword in reason for keyword in ['suffix confusion', 'ending confusion', '-ent vs -ant', 'ending sounds same']):
        error_category = "Phonetic"
        error_subcategory = "Suffix"

    # Orthographic patterns (only for complex/specialized terms)
    elif any(keyword in reason for keyword in ['professional term', 'technical term', 'domain specific', 'specialized vocabulary', 'medical term', 'beauty industry term', 'complex word', 'unfamiliar spelling']):
        error_category = "Orthographic"
        error_subcategory = "Hard Word"

    # If no category matched exactly, this error doesn't fit our strict classification system
    if not error_category or not error_subcategory:
        return None

    return {
        'error_category': error_category,
        'error_subcategory': error_subcategory,
        'error_explanation': f"Classification based on reason analysis: {error.get('reason', '')}"
    }


def validate_error_classification(error):
    """Classify error into exactly one category and subcategory based on reason analysis."""
    return classify_error_from_reason(error)


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


def print_error_details(error, prefix="❌", start_time=0):
    """Print standardized error details."""
    category = error.get('error_category', 'unknown')
    subcategory = error.get('error_subcategory', 'unknown')
    explanation = error.get('error_explanation', 'no explanation')
    log_message(f"{prefix} Error in sentence {error['sentence_idx'] + 1}: '{error['word']}' → '{error['correct']}'", start_time)
    log_message(f"   Category: {category} | Subcategory: {subcategory} | Explanation: {explanation}", start_time)


def create_spelling_analysis_prompt(sentence):
    """Create unified spelling analysis prompt for both passes."""
    return f"""<s> [INST] ## Task: Spelling Error Analysis and Classification

**Input Sentence:**
"{sentence}"

**Analysis Requirements:**
You are an expert proofreader specializing in Amazon beauty product reviews. Analyze this sentence for genuine spelling errors and classify each error according to the mandatory error classification system.

### CRITICAL RESTRICTIONS (ZERO TOLERANCE):
🚫 **NO CAPITALIZATION**: Never report case differences ('i' vs 'I', 'regimen' vs 'Regimen').
🚫 **NO GRAMMAR ERRORS**: Do not report ANY grammar issues including:
  - Subject-verb agreement (she like → she likes)
  - Singular/plural errors (every months → every month)
  - Tense conjugation (is → was, have → had)
  - Preposition errors (at → in, on → with)
  - Conjunction issues (and → but, or → nor)
  - Article usage (a → the, an → a)
  - Word order or sentence structure problems
  - Pronoun errors (its/it is/it has confusion)
  - Possessive vs contraction errors (its vs it's, your vs you're)
  - Auxiliary/modal verb errors
  - Any morphological or syntactic corrections
🚫 **NO SPACING ERRORS**: Do not report missing/extra spaces between words, including:
  - Compound words (granddaughter → grand daughter)
  - Word contractions (cannot → can not)
  - Any spacing/formatting issues between words
🚫 **NO PUNCTUATION**: Ignore missing/extra periods, commas, question marks, etc.
🚫 **NO FORMATTING**: Ignore spacing, extra periods, or formatting problems.
🚫 **NO STYLE POLICING**: Valid words like "faux", "literally", "cute" are acceptable even if informal.

### AMAZON & DOMAIN SPECIFIC RULES:
- **AMAZON CONTEXT**: Terms like "Prime", "Subscribe & Save" are valid platform-specific language.
- **DOMAIN EXPERTISE**: Beauty/skin care terms are correct including:
  - **Fabric/Textile Terms**: "cloths" (face wipes, microfiber cloths), "wash cloths", "microfiber", "flannel", "terry cloth", "cotton rounds", "towelettes"
  - **Product Categories**: "face wipes", "cleansing cloths", "makeup remover cloths", "eyelash curler", "water flosser", "emery board", "nail buffer"
  - **Skin Care Terms**: "moisturizer", "cleanser", "toner", "serum", "hyaluronic acid", "retinol", "vitamin C", "peptide", "hyaluronic", "ceramide"
  - **Beauty Tools**: "foundation brush", "beauty blender", "concealer brush", "eyeshadow palette", "lip liner", "mascara wand"
- **FABRIC vs CLOTHING DISTINCTION**: In beauty reviews, "cloths" typically refers to cleansing/face wipes, not clothing items. Only correct if context clearly indicates garments.
- **BRAND PROTECTION**: Respect potential brand names (e.g., "Iryasa", "Isntree", "The Ordinary"). If unsure, favor "No Error".

### ERROR CLASSIFICATION SYSTEM (MANDATORY):
Each reported error MUST be classified into ONE SPECIFIC SUBCATEGORY from the following system. If an error doesn't fit any subcategory exactly, DO NOT report it.

**1. 机械性错误 (Mechanical/Typo)**
- **漏输 (Deletion)**: Random input errors where letters are missing due to incomplete key presses or keyboard unresponsiveness. Common in consonants or final vowels.
- **多输 (Insertion)**: Extra letters due to key sticking or accidental adjacent key presses, often repeating letters or inserting irrelevant characters.
- **字母换位 (Transposition)**: Adjacent letters swapped due to uncoordinated typing rhythm or brief encoding errors in the brain.
- **复杂混淆 (Scramble)**: Multiple mechanical errors combined, creating chaotic middle sections in longer words during editing.

**2. 语音/听觉错误 (Phonetic)**
- **同音/形近词 (Homophone)**: Brain retrieves wrong spelling entry based on pronunciation. Occurs with identical-sounding words where meaning connection is weak.
- **后缀混淆 (Suffix)**: Same-sounding but differently spelled endings (e.g., -ent/-ant). User lacks root knowledge and chooses by pronunciation instinct.

**3. 正字法/认知错误 (Orthographic)**
- **复杂词汇 (Hard Word)**: Domain-specific professional terms, academic vocabulary, or loanwords. User knows the word but lacks accurate spelling memory, resulting in intuitive but incorrect spelling.

### ANALYSIS CRITERIA:
Only report genuine spelling mistakes that fit the above classification system. Focus on:
- **Spelling Errors**: Wrong/missing letters (e.g., "regimine" → "regimen", "moistutrizer" → "moisturizer")
- **Context Mix-ups**: ONLY homophones or near-homophones that are genuine typos
- **Technical Terms**: Beauty/skin care terminology errors with actual spelling mistakes

### OUTPUT FORMAT (Strict JSON):
Return ONLY genuine spelling errors. If no errors found, return {{"error_details": []}}.

For each error, provide:
- `word_position`: Approximate word position in sentence (0-based)
- `original_word`: The misspelled word
- `corrected_word`: The correct spelling
- `reason`: Detailed explanation of why this is a spelling error and what type of error it is

{{
  "error_details": [
    {{
      "word_position": 5,
      "original_word": "regimine",
      "corrected_word": "regimen",
      "reason": "Missing 'e' in 'regimen' - this appears to be a typing error where the 'e' was omitted during fast typing of this beauty product term"
    }}
  ]
}}
[/INST]"""


def handle_llm_error(response_str, context="", start_time=0):
    """Handle LLM response parsing errors with simplified diagnostics."""
    log_message(f"⚠️ JSON parse error in {context}: response format issue", start_time)
    log_message(f"📊 Response length: {len(response_str)} characters", start_time)

    if '[INST]' in response_str:
        log_message("🔍 Issue: LLM returned prompt instead of JSON", start_time)
    elif '{' not in response_str and '[' not in response_str:
        log_message("🔍 Issue: No JSON structure found", start_time)
    else:
        log_message("🔍 Issue: Malformed JSON structure", start_time)


# ============================================================================
# Main Processing Functions
# ============================================================================








def analyze_single_sentence_second_pass(sentence, sentence_idx, llm_model, start_time):
    """Perform second-pass analysis on sentences that had no errors in first pass."""
    try:
        print(f"[{time.time() - start_time:.1f}s] 🔄 Second-pass analyzing sentence {sentence_idx + 1}: {sentence[:100]}{'...' if len(sentence) > 100 else ''}", flush=True)
        sys.stdout.flush()
    except Exception as print_e:
        print(f"[{time.time() - start_time:.1f}s] ⚠️ Failed to print sentence {sentence_idx + 1}: {print_e}", flush=True)
        sys.stdout.flush()

    prompt = create_spelling_analysis_prompt(sentence)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model.invoke(messages)
        response_str = response.content.strip()

        # Parse JSON response using unified function
        json_str = parse_llm_json_response(response_str)

        errors = []
        if json_str:
            try:
                result = json.loads(json_str)
                if isinstance(result, dict) and 'error_details' in result:
                    error_list = result['error_details']
                elif isinstance(result, list):
                    error_list = result
                else:
                    error_list = []

                for error in error_list:
                    if isinstance(error, dict) and 'original_word' in error and 'corrected_word' in error:
                        # Validate error classification using unified function
                        validation_result = validate_error_classification(error)
                        if validation_result:
                            errors.append({
                                'sentence_idx': sentence_idx,
                                'word_position': error.get('word_position', -1),
                                'word': error['original_word'],
                                'correct': error['corrected_word'],
                                'sentence': sentence,
                                **validation_result
                            })

            except json.JSONDecodeError:
                handle_llm_error(response_str, f"second-pass sentence {sentence_idx + 1}", start_time)
                return []

        return errors

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed to analyze sentence {sentence_idx + 1} (second pass): {str(e)[:50]}...", flush=True)
        return []


def analyze_single_sentence(sentence, sentence_idx, llm_model, start_time):
    """Analyze a single sentence with LLM."""
    import sys

    # COMMENTED OUT: Individual sentence processing logs
    # try:
    #     print(f"[{time.time() - start_time:.1f}s] 📝 Analyzing sentence {sentence_idx + 1}: {sentence[:100]}{'...' if len(sentence) > 100 else ''}", flush=True)
    #     sys.stdout.flush()
    # except Exception as print_e:
    #     print(f"[{time.time() - start_time:.1f}s] ⚠️ Failed to print sentence {sentence_idx + 1}: {print_e}", flush=True)
    #     sys.stdout.flush()

    prompt = create_spelling_analysis_prompt(sentence)

    try:
        response_str, success = call_llm_with_retry(llm_model, prompt, max_retries=3, context=f"sentence {sentence_idx + 1}")

        if not success:
            print(f"[{time.time() - start_time:.1f}s] ⚠️ LLM call failed for sentence {sentence_idx + 1} after retries", flush=True)
            return []

        # Parse JSON response using unified function
        json_str = parse_llm_json_response(response_str)

        errors = []
        if json_str:
            try:
                result = json.loads(json_str)
                # New format: {"error_details": [...]}
                if isinstance(result, dict) and 'error_details' in result:
                    error_list = result['error_details']
                elif isinstance(result, list):
                    error_list = result
                else:
                    error_list = []

                for error in error_list:
                    # Handle sentence-level analysis format
                    if isinstance(error, dict) and 'original_word' in error and 'corrected_word' in error:
                        # Validate error classification using unified function
                        validation_result = validate_error_classification(error)
                        if validation_result:
                            errors.append({
                                'sentence_idx': sentence_idx,
                                'word_position': error.get('word_position', -1),
                                'word': error['original_word'],
                                'correct': error['corrected_word'],
                                'sentence': sentence,
                                **validation_result
                            })

            except json.JSONDecodeError:
                handle_llm_error(response_str, f"sentence {sentence_idx + 1}", start_time)
                return []

        return errors

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed to analyze sentence {sentence_idx + 1}: {str(e)[:50]}...", flush=True)
        sys.stdout.flush()
        return []


def tokenize_words(sentence, nlp):
    """Tokenize words using spaCy for proper handling of English text."""
    doc = nlp(sentence)

    words = []
    for token in doc:
        # Include tokens that are:
        # - Alphabetic (words)
        # - Contain apostrophes (contractions like "don't")
        # - Hyphenated compounds
        # Exclude pure punctuation like ".", ",", etc.
        if token.is_alpha or "'" in token.text or "-" in token.text:
            words.append(token.text)

    return words


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

✅ **SHOULD BE CONFIRMED** (return true) for properly classified errors that fit these categories:
- **Mechanical/Typo**: Random typing errors (Deletion, Insertion, Transposition, Scramble)
- **Phonetic**: Sound-based errors (Homophone, Suffix confusion)
- **Orthographic**: Knowledge-based errors (Hard Word spelling difficulties)

❌ **SHOULD BE REJECTED** (return false) for:
- Incorrect or invalid error classifications
- Errors that don't fit any of the three main categories
- Grammar issues (subject-verb agreement, tense, prepositions, etc.)
- Spacing/punctuation/capitalization issues
- Style preferences or informal language choices
- Any corrections that are actually grammatical rather than spelling-based

**Decision Rules:**
- Verify that the error genuinely fits the assigned category and subcategory
- Be extremely conservative - only confirm errors with proper classification
- Amazon reviews are informal - respect colloquial language and domain terminology
- Beauty industry has specialized terms - don't flag legitimate domain vocabulary as errors

Return your decision as JSON:
{{
  "is_valid_error": true/false,
  "confidence_level": "high/medium/low",
  "validation_reason": "brief explanation of your decision"
}}
[/INST]"""

            response_str, success = call_llm_with_retry(llm_model, validation_prompt, max_retries=3, context=f"validation error {error_idx + 1}")

            if not success:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ Validation LLM call failed for error {error_idx + 1} after retries", flush=True)
                if attempt < max_retries - 1:
                    continue
                else:
                    return error, True

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
                        print(f"[{time.time() - start_time:.1f}s]    ✅ Confirmed (confidence: {confidence})", flush=True)
                        return error, True
                    else:
                        word = error.get('word', 'unknown')
                        correct = error.get('correct', 'unknown')
                        print(f"[{time.time() - start_time:.1f}s]    ❌ Rejected (confidence: {confidence}) '{word}' → '{correct}' - {reason}", flush=True)
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
        reason_validation_prompt = f"""<s> [INST] ## Task: Validate Error Classification Explanation

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
✅ **ACCURATE** if the explanation correctly describes:
- **Mechanical/Typo**: Random typing errors, keyboard issues, or input mistakes
- **Phonetic**: Sound-based retrieval errors or pronunciation confusions
- **Orthographic**: Knowledge gaps in spelling complex or domain-specific terms

❌ **INACCURATE** if the explanation:
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

[/INST]"""

        response_str, success = call_llm_with_retry(llm_model, reason_validation_prompt, max_retries=3, context="reason validation")

        if not success:
            print(f"[{time.time() - start_time:.1f}s]    ⚠️ Reason validation LLM call failed after retries", flush=True)
            return True, 'unknown', None, 'LLM call failed'

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
    max_validation_concurrent = min(5, len(all_errors))

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

    print(f"[{time.time() - start_time:.1f}s] ✅ Comprehensive validation complete: {len(validated_errors)}/{len(all_errors)} errors confirmed, {explanation_corrections} explanations improved", flush=True)
    sys.stdout.flush()

    return validated_errors


def analyze_text_with_llm(text, llm_model, start_time=0):
    """Split text into sentences and analyze with concurrent LLM requests."""
    import sys

    # Preprocess text: replace hyphens with spaces to avoid tokenization issues
    text = text.replace("-", " ")
    print(f"[{time.time() - start_time:.1f}s] 📝 Preprocessed text (hyphens replaced with spaces)", flush=True)

    # Split text into sentences
    print(f"[{time.time() - start_time:.1f}s] 📝 Splitting text into sentences...", flush=True)
    sys.stdout.flush()

    sentences = sent_tokenize(text)

    print(f"[{time.time() - start_time:.1f}s] 📝 Split into {len(sentences)} sentences", flush=True)
    sys.stdout.flush()

    all_errors = []
    processed_sentences = 0
    sentences_with_errors = set()  # Track which sentences had errors

    # FIRST PASS: Process sentences with concurrent LLM requests (max 50 concurrent)
    max_concurrent = 50

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all sentence analysis tasks
        future_to_sentence = {}
        for sent_idx, sentence in enumerate(sentences):
            future = executor.submit(analyze_single_sentence, sentence.strip(), sent_idx, llm_model, start_time)
            future_to_sentence[future] = (sent_idx, sentence)

        # Process completed tasks as they finish
        for future in as_completed(future_to_sentence):
            sent_idx, sentence = future_to_sentence[future]
            processed_sentences += 1

            try:
                errors = future.result()
                if errors:
                    all_errors.extend(errors)
                    sentences_with_errors.add(sent_idx)  # Mark this sentence as having errors

                    # # COMMENTED OUT: Print error details during analysis
                    # for error in errors:
                    #     print_error_details(error, "❌", start_time)

            except Exception as e:
                print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed to get result for sentence {sent_idx + 1}: {str(e)[:50]}...", flush=True)
                sys.stdout.flush()

            # Show progress
            if processed_sentences % 10 == 0 or processed_sentences == len(sentences):
                print(f"[{time.time() - start_time:.1f}s] 📊 Progress: {processed_sentences}/{len(sentences)} sentences processed", flush=True)
                sys.stdout.flush()

    print(f"[{time.time() - start_time:.1f}s] 📊 First pass complete: {processed_sentences} sentences processed, found {len(all_errors)} spelling errors", flush=True)
    sys.stdout.flush()

    # # COMMENTED OUT: Second pass analysis
    # # Wait 1 minute before starting second pass
    # print(f"[{time.time() - start_time:.1f}s] ⏳ Waiting 1 minute before starting second pass...", flush=True)
    # sys.stdout.flush()
    # time.sleep(60)
    #
    # # SECOND PASS: Analyze sentences that had no errors in first pass
    # sentences_without_errors = [(idx, sent) for idx, sent in enumerate(sentences) if idx not in sentences_with_errors]
    #
    # if sentences_without_errors:
    #     print(f"[{time.time() - start_time:.1f}s] 🔄 Starting second-pass analysis of {len(sentences_without_errors)} sentences that had no errors...", flush=True)
    #     sys.stdout.flush()
    #
    #     second_pass_errors = []
    #
    #     with ThreadPoolExecutor(max_workers=min(5, len(sentences_without_errors))) as executor:
    #         # Submit second-pass analysis tasks
    #         future_to_second_pass = {}
    #         for sent_idx, sentence in sentences_without_errors:
    #             future = executor.submit(analyze_single_sentence_second_pass, sentence.strip(), sent_idx, llm_model, start_time)
    #             future_to_second_pass[future] = (sent_idx, sentence)
    #
    #         # Process second-pass results
    #         for future in as_completed(future_to_second_pass):
    #             sent_idx, sentence = future_to_second_pass[future]
    #
    #             try:
    #                 errors = future.result()
    #                 if errors:
    #                     second_pass_errors.extend(errors)
    #
    #                     # Print details of each error found in second pass
    #                     for error in errors:
    #                         print_error_details(error, "🔄", start_time)
    #
    #             except Exception as e:
    #                 print(f"[{time.time() - start_time:.1f}s]    ⚠️ Failed second-pass analysis for sentence {sent_idx + 1}: {str(e)[:50]}...", flush=True)
    #                 sys.stdout.flush()
    #
    #     if second_pass_errors:
    #         all_errors.extend(second_pass_errors)
    #         print(f"[{time.time() - start_time:.1f}s] ✅ Second-pass analysis found {len(second_pass_errors)} additional errors", flush=True)
    #     else:
    #         print(f"[{time.time() - start_time:.1f}s] ✅ Second-pass analysis found no additional errors", flush=True)
    #
    #     sys.stdout.flush()
    #
    # print(f"[{time.time() - start_time:.1f}s] 📊 Total errors found: {len(all_errors)} (first pass: {len(all_errors) - len(second_pass_errors or [])}, second pass: {len(second_pass_errors or [])})", flush=True)
    # sys.stdout.flush()

    # Initialize second_pass_errors for compatibility
        second_pass_errors = []

    print(f"[{time.time() - start_time:.1f}s] 📊 Total errors found: {len(all_errors)} (first pass only)", flush=True)
                    sys.stdout.flush()

    # # COMMENTED OUT: Wait before validation
    # if all_errors:
    #     print(f"[{time.time() - start_time:.1f}s] ⏳ Waiting 1 minute before starting error validation...", flush=True)
    #     sys.stdout.flush()
    #     time.sleep(60)

    # Secondary validation of errors
    if all_errors:
        print(f"[{time.time() - start_time:.1f}s] 🔍 Starting secondary validation of {len(all_errors)} errors...", flush=True)
        sys.stdout.flush()
        validated_errors = validate_errors_with_llm(all_errors, llm_model, start_time)
        print(f"[{time.time() - start_time:.1f}s] ✅ Secondary validation complete: {len(validated_errors)}/{len(all_errors)} errors confirmed", flush=True)
        sys.stdout.flush()

        # Print details of all confirmed errors
        if validated_errors:
            print(f"[{time.time() - start_time:.1f}s] 📋 Final confirmed spelling errors:", flush=True)
            sys.stdout.flush()
            for error in validated_errors:
                print_error_details(error, "✅", start_time)

        return validated_errors

    return all_errors




def process_users(input_file, output_file, llm_model, start_time=0):
    """Process users: load JSON file and select user with >100 reviews for analysis."""
    import sys
    import time
    print(f"[{time.time() - start_time:.1f}s] Loading data from: {input_file}", flush=True)
    sys.stdout.flush()

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"[{time.time() - start_time:.1f}s] ❌ Input file does not exist: {input_file}", flush=True)
        sys.stdout.flush()
        return None

    try:
        print(f"[{time.time() - start_time:.1f}s] 📖 Loading JSON file...", flush=True)
        sys.stdout.flush()

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find users with more than 100 reviews
        qualified_users = [(user_id, user_data['review_count']) for user_id, user_data in data.items()
                          if user_data['review_count'] > 100]

        if not qualified_users:
            print(f"[{time.time() - start_time:.1f}s] ❌ No users found with more than 100 reviews", flush=True)
            sys.stdout.flush()
            return None

        # Select user with most reviews
        qualified_users.sort(key=lambda x: x[1], reverse=True)
        selected_user_id, review_count = qualified_users[0]

        print(f"[{time.time() - start_time:.1f}s] 👤 Selected user {selected_user_id} with {review_count} reviews", flush=True)
                        sys.stdout.flush()

        # Extract all review texts for the selected user
        user_data = data[selected_user_id]
        user_reviews = []

        for review in user_data['reviews']:
            text = review.get('review_text', '').strip()
            if text:
                user_reviews.append(text)

            if not user_reviews:
            print(f"[{time.time() - start_time:.1f}s] ❌ No review texts found for user {selected_user_id}", flush=True)
                sys.stdout.flush()
                return None

            # Combine all reviews for this user
        max_reviews = len(user_reviews)
        selected_reviews = user_reviews
        combined_text = ' '.join(selected_reviews)

        print(f"[{time.time() - start_time:.1f}s] ✅ Using {max_reviews}/{len(user_reviews)} reviews for user {selected_user_id}", flush=True)
            print(f"[{time.time() - start_time:.1f}s] 📏 Combined text length: {len(combined_text)} characters", flush=True)
            sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Error reading/parsing file: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return None

    # Analyze with LLM
    print(f"[{time.time() - start_time:.1f}s] 🤖 Analyzing text with LLM (all sentences)...", flush=True)
    sys.stdout.flush()

    try:
        errors = analyze_text_with_llm(combined_text, llm_model, start_time)
        print(f"[{time.time() - start_time:.1f}s] LLM analysis completed, found {len(errors)} errors", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ LLM analysis failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None

    # Save results
    result = {
        'user_id': selected_user_id,
        'review_count': len(user_reviews),
        'reviews_used': max_reviews,
        'text_length': len(combined_text),
        'spelling_errors': errors,
        'error_count': len(errors),
        'sample_reviews': selected_reviews[:3]  # Save first 3 used reviews as sample
    }

    try:
        print(f"[{time.time() - start_time:.1f}s] 💾 Saving results...", flush=True)
        sys.stdout.flush()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{time.time() - start_time:.1f}s] ✅ Results saved to: {output_file}", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[{time.time() - start_time:.1f}s] ❌ Failed to save results: {e}", flush=True)
        sys.stdout.flush()
        return None

    print(f"[{time.time() - start_time:.1f}s] 🎉 Found {len(errors)} spelling errors across {max_reviews} reviews (from user with {len(user_reviews)} total reviews)", flush=True)
    sys.stdout.flush()

    return result


def test_sentences():
    """Test sentence tokenization."""
    import sys

    # Initialize spaCy for testing
    try:
        test_nlp = spacy.load("en_core_web_sm")
    except OSError:
        print(f"❌ Failed to load spaCy model for testing", flush=True)
        return
    except Exception as e:
        print(f"❌ Failed to load spaCy model for testing: {e}", flush=True)
        return

    test_sentences = [
        "It actually is an entire skin care regimine containing a cleanser.",
        "Typically, my skin care routine uses hyalauronic serum."
    ]

    print("Testing sentence tokenization:", flush=True)
    sys.stdout.flush()
    for i, sentence in enumerate(test_sentences, 1):
        words = tokenize_words(sentence, test_nlp)
        print(f"Sentence {i}: {len(words)} words", flush=True)
        sys.stdout.flush()


def main():
    """Main function."""
    import sys
    import time

    start_time = time.time()
    print(f"🚀 [{time.time() - start_time:.1f}s] Starting program...", flush=True)
    sys.stdout.flush()

    print(f"📍 [{time.time() - start_time:.1f}s] Initializing SiliconFlow model...", flush=True)
    sys.stdout.flush()

    try:
        llm_model = get_gm_model()
        print(f"✅ [{time.time() - start_time:.1f}s] SiliconFlow model initialized successfully", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to initialize model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # Test sentence tokenization
    print(f"🧪 [{time.time() - start_time:.1f}s] Testing sentence tokenization...", flush=True)
    sys.stdout.flush()
    test_sentences()

    # Simple processing
    print(f"📁 [{time.time() - start_time:.1f}s] Processing file: {INPUT_FILE}", flush=True)
    print(f"💾 [{time.time() - start_time:.1f}s] Output file: {OUTPUT_FILE}", flush=True)
    sys.stdout.flush()

    result = process_users(INPUT_FILE, OUTPUT_FILE, llm_model, start_time)

    print(f"🏁 [{time.time() - start_time:.1f}s] Program completed", flush=True)
    sys.stdout.flush()

    return result


if __name__ == "__main__":
    main()

