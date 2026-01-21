#!/usr/bin/env python3
"""
Spelling Error Style Analysis using SiliconFlow Batch API

This script analyzes spelling errors in Amazon beauty product reviews using
SiliconFlow's batch inference API for cost-effective processing.

Features:
- Batch processing with Qwen/QwQ-32B model
- 50% cost reduction compared to real-time API
- Asynchronous processing (up to 24 hours)
- Handles large-scale text analysis
"""

import os
import json
import time
import sys
import requests
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# Configuration
SILICONFLOW_API_KEY = "sk-drezmfyckjkmxixpiblvbwdhypjbrsoyvmeertajtupiqnnj"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/QwQ-32B"
INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json"
BATCH_OUTPUT_DIR = "/home/wlia0047/ar57_scratch/wenyu/batch_results"

# Initialize OpenAI client for SiliconFlow
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=BASE_URL
)

def create_spelling_analysis_prompt(review_text: str) -> str:
    return f"""<s> [INST] ## Task: Spelling Error Classification (Arts & Crafts Domain)

**Input:** "{review_text}"

**Goal:** Identify GENUINE spelling errors (typos) in this ENTIRE review text and classify them. Return JSON only.

### 🚫 IGNORE (DO NOT REPORT):
1. **Grammar/Syntax:** Subject-verb agreement, tense, prepositions, plurals (e.g., "she like" -> IGNORE).
2. **Punctuation/Spacing:** Missing commas, periods, or spaces.
3. **Capitalization:** Case differences (e.g., "amazon" vs "Amazon").
4. **Brand Names:** If you think this might be a brand name, do not make any spelling error judgment. Skip it entirely.

### 📂 CATEGORIES:
1. **Deletion** (Character REMOVED)
   - Definition: One or more letters are missing from the word (forgotten keystrokes).
   - Example: "wit" -> "with" (missing 'h'), "colr" -> "color" (missing 'o').
2. **Insertion** (Character ADDED)
   - Definition: Extra letters are inserted into the word (accidental keystrokes).
   - Example: "recieve" -> "receive" (extra 'e'), "definately" -> "definitely" (extra 'a').
3. **Transposition** (Letters SWAPPED)
   - Definition: Two adjacent letters are swapped in position (finger slip on keyboard).
   - Example: "teh" -> "the", "wierd" -> "weird", "seperate" -> "separate".
4. **Scramble** (Complex REARRANGEMENT)
   - Definition: Multiple letters are rearranged in a complex way (not just adjacent swaps).
   - Example: "reccomend" -> "recommend", "acheive" -> "achieve", "pronounciation" -> "pronunciation".
5. **Substitution** (Character Count SAME, Identity CHANGED)
   - Definition: A specific letter is replaced by a WRONG letter at the SAME position. The total word length remains unchanged.
   - Common Causes:
     1. **Fat Finger:** Hitting a nearby key (e.g., 'p' instead of 'o').
     2. **Phonetic Guess:** Using 'k' instead of 'c', or 'z' instead of 's'.
     3. **Vowel Confusion:** Using 'a' instead of 'e' (schwa sound).
   - 🚫 DISTINCTION: If letters are swapped (le -> el), that is Transposition, NOT Substitution.
   - Example:
     - "curlique" -> "curlicue" ('q' replaced 'c' -> Phonetic)
     - "definatly" -> "definitely" ('a' replaced 'i' -> Vowel)
     - "wprk" -> "work" ('p' replaced 'o' -> Keyboard Slip)
6. **Homophone** (Sound-alike WORDS)
   - Definition: Words that sound the same but are spelled differently (confusion between similar-sounding words).
   - Example: "there" -> "their", "to" -> "too", "its" -> "it's".
7. **Suffix** (Ending WRONG)
   - Definition: The suffix or ending of the word is incorrect (common in irregular forms).
   - Example: "runing" -> "running", "begining" -> "beginning", "embosing" -> "embossing".
8. **Hard Word** (Difficult SPELLING)
   - Definition: Words with unusual or complex spelling patterns that are frequently misspelled.
   - Example: "embosing" -> "embossing", "rythm" -> "rhythm", "acheive" -> "achieve".
9. **Extra Space** (Unnecessary SPACE)
   - Definition: Compound words that are normally written together, but are incorrectly separated by spaces.
   - Example: "card stock" -> "cardstock", "hot dog" -> "hotdog", "high school" -> "highschool".
10. **Extra Hyphen** (Unnecessary HYPHEN)
    - Definition: Compound words that are normally written together, but are incorrectly separated by hyphens.
    - Example: "mother-in-law" -> "motherinlaw", "self-control" -> "selfcontrol", "state-of-the-art" -> "stateoftheart".

### 💡 EXAMPLES (Few-Shot):

Input: "I used my Cuttlebug to cut the felt."
Output: {{ "Deletion": [], "Insertion": [], "Transposition": [], "Scramble": [], "Substitution": [], "Homophone": [], "Suffix": [], "Hard Word": [], "Extra Space": [], "Extra Hyphen": [] }}

Input: "The colr was fadeing fast."
Output: {{
  "Deletion": [{{ "original_word": "colr", "corrected_word": "color", "reason": "Missing 'o'" }}],
  "Substitution": [],
  "Suffix": [{{ "original_word": "fadeing", "corrected_word": "fading", "reason": "Bad suffix" }}],
  "Extra Space": [],
  "Extra Hyphen": []
}}

Input: "I love my Grand Calibur machine."
Output: {{ "Deletion": [], "Insertion": [], "Transposition": [], "Scramble": [], "Substitution": [], "Homophone": [], "Suffix": [], "Hard Word": [], "Extra Space": [], "Extra Hyphen": [] }}

Input: "I bought card stock and mother-in-law for my project."
Output: {{
  "Deletion": [],
  "Insertion": [],
  "Transposition": [],
  "Scramble": [],
  "Substitution": [],
  "Homophone": [],
  "Suffix": [],
  "Hard Word": [],
  "Extra Space": [{{ "original_word": "card stock", "corrected_word": "cardstock", "reason": "Compound word should be written together" }}],
  "Extra Hyphen": [{{ "original_word": "mother-in-law", "corrected_word": "motherinlaw", "reason": "Compound word should be written together" }}]
}}

### OUTPUT JSON:
Return ONLY the JSON object. If no errors, return empty arrays.
[/INST]"""



def _is_genuine_spelling_error(original: str, corrected: str, reason: str) -> bool:
    """
    Validate if this is a genuine spelling error vs grammar/morphology/style issue.

    Args:
        original: Original word
        corrected: Corrected word
        reason: Reason for correction

    Returns:
        True if this is a genuine spelling error
    """
    # Must have actual character differences
    if original == corrected:
        return False

    # Check for character-level differences (insertions, deletions, substitutions, transpositions)
    # This is a simplified check - real implementation would use edit distance algorithms
    original_chars = original.lower()
    corrected_chars = corrected.lower()

    # Allow for reasonable edit distance (1-3 character changes for genuine spelling errors)
    if abs(len(original_chars) - len(corrected_chars)) > 3:
        return False  # Too different, likely not spelling error

    # Check if it's a common morphology change (add/remove 's' for plural)
    if original_chars + 's' == corrected_chars or corrected_chars + 's' == original_chars:
        return False  # Likely plural/singular morphology, not spelling

    # Check if it's verb form change (add/remove 's', 'ed', 'ing', etc.)
    verb_suffixes = ['s', 'ed', 'ing', 'er', 'est']
    for suffix in verb_suffixes:
        if original_chars + suffix == corrected_chars or corrected_chars + suffix == original_chars:
            return False  # Likely verb conjugation, not spelling

    return True  # Passed basic validation


def prepare_batch_requests(sentences: List[str], user_id: str) -> List[Dict[str, Any]]:
    """
    Prepare batch requests in JSONL format for SiliconFlow batch API.

    Args:
        sentences: List of reviews to analyze (whole reviews, not split into sentences)
        user_id: User identifier for tracking

    Returns:
        List of request dictionaries for batch processing
    """
    batch_requests = []

    for idx, sentence in enumerate(sentences):
        custom_id = f"{user_id}_review_{idx}"

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": create_spelling_analysis_prompt(sentence)
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "stream": False,
                "thinking_budget": 32768
            }
        }
        batch_requests.append(request)

    return batch_requests

def prepare_batch_requests_combined(sentences: List[str], user_mapping: List[tuple]) -> List[Dict[str, Any]]:
    """
    Prepare batch requests for combined reviews from multiple users.

    Args:
        sentences: List of all reviews to analyze
        user_mapping: List of (user_id, local_review_idx) tuples

    Returns:
        List of request dictionaries for batch processing
    """
    batch_requests = []

    for global_idx, (sentence, (user_id, local_idx)) in enumerate(zip(sentences, user_mapping)):
        custom_id = f"{user_id}_review_{local_idx}_global_{global_idx}"

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": create_spelling_analysis_prompt(sentence)
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "stream": False,
                "thinking_budget": 32768
            }
        }
        batch_requests.append(request)

    return batch_requests

def save_batch_file(batch_requests: List[Dict[str, Any]], filename: str) -> str:
    """
    Save batch requests to JSONL file.

    Args:
        batch_requests: List of request dictionaries
        filename: Output filename

    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f"✅ Batch file saved: {filename} ({len(batch_requests)} requests)", flush=True)
    return filename

def upload_batch_file(file_path: str) -> str:
    """
    Upload batch file to SiliconFlow.

    Args:
        file_path: Path to batch file

    Returns:
        File ID from SiliconFlow, or None if upload failed
    """
    print(f"📤 Uploading batch file: {file_path}", flush=True)

    try:
        with open(file_path, "rb") as f:
            batch_file = client.files.create(
                file=f,
                purpose="batch"
            )

        # Get file ID from the data attribute
        file_id = batch_file.data.get('id')
        if not file_id:
            print("❌ File upload response missing ID", flush=True)
            return None

        print(f"✅ File uploaded successfully. File ID: {file_id}", flush=True)
        return file_id

    except Exception as e:
        print(f"❌ Failed to upload batch file: {e}", flush=True)
        return None

def create_batch_job(file_id: str, batch_name: str) -> str:
    """
    Create batch inference job.

    Args:
        file_id: SiliconFlow file ID
        batch_name: Name for the batch job

    Returns:
        Batch job ID, or None if creation failed
    """
    print(f"🚀 Creating batch job: {batch_name}", flush=True)

    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Spelling error analysis - {batch_name}",
                "model": MODEL_NAME,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            extra_body={"replace": {"model": MODEL_NAME}}
        )

        batch_id = batch.id
        print(f"✅ Batch job created. Batch ID: {batch_id}", flush=True)
        return batch_id

    except Exception as e:
        print(f"❌ Failed to create batch job: {e}", flush=True)
        return None

def monitor_batch_job(batch_id: str) -> Dict[str, Any]:
    """
    Monitor batch job status until completion.

    Args:
        batch_id: Batch job ID

    Returns:
        Final batch status
    """
    print(f"👀 Monitoring batch job: {batch_id}", flush=True)

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        # Handle cases where request_counts might be None
        if batch.request_counts:
            completed = getattr(batch.request_counts, 'completed', 0)
            total = getattr(batch.request_counts, 'total', 0)
            progress_str = f"{completed}/{total}"
        else:
            progress_str = "pending"

        print(f"📊 Status: {status} | Progress: {progress_str}", flush=True)

        if status in ['completed', 'failed', 'expired', 'cancelled']:
            print(f"🏁 Batch job finished with status: {status}", flush=True)
            return batch.__dict__

        time.sleep(60)  # Check every minute

def download_batch_results(batch_info: Dict[str, Any], output_file: str) -> str:
    """
    Download batch results.

    Args:
        batch_info: Batch job information
        output_file: Output file path

    Returns:
        Path to downloaded results file
    """
    if 'output_file_id' not in batch_info or not batch_info['output_file_id']:
        print("❌ No output file available", flush=True)
        return None

    output_file_url = batch_info['output_file_id']
    print(f"📥 Downloading results from: {output_file_url}", flush=True)

    try:
        # Check if it's a URL (starts with http)
        if output_file_url.startswith('http'):
            # Download directly from URL with authentication
            import requests

            headers = {
                'Authorization': f'Bearer {SILICONFLOW_API_KEY}',
                'User-Agent': 'Mozilla/5.0 (compatible; BatchDownload/1.0)'
            }

            print(f"📥 Downloading from URL with authentication...", flush=True)
            response = requests.get(output_file_url, headers=headers, timeout=60)
            response.raise_for_status()

            # Save to local file
            with open(output_file, 'wb') as f:
                f.write(response.content)

            print(f"✅ Downloaded {len(response.content)} bytes", flush=True)
        else:
            # Use OpenAI client for file ID
            print(f"📥 Downloading using file ID...", flush=True)
            file_content = client.files.content(output_file_url)
            with open(output_file, 'wb') as f:
                f.write(file_content.content)

        print(f"✅ Results saved to: {output_file}", flush=True)
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error downloading results: {e}", flush=True)
        return None
    except Exception as e:
        print(f"❌ Failed to download results: {e}", flush=True)
        return None

def process_batch_results(results_file: str, sentences: List[str] = None) -> List[Dict[str, Any]]:
    """
    Process batch results and extract spelling errors.

    Args:
        results_file: Path to batch results file
        sentences: List of original reviews (for review lookup by index)

    Returns:
        List of processed spelling errors
    """
    print(f"🔍 Processing batch results: {results_file}", flush=True)

    all_errors = []

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())

            if 'response' in result and 'body' in result['response']:
                response_body = result['response']['body']

                if 'choices' in response_body and response_body['choices']:
                    content = response_body['choices'][0]['message']['content']

                    # Parse JSON response - handle multiple JSON objects in response
                    try:
                        json_start = content.find('{')
                        if json_start != -1:
                            # Find the first complete JSON object by counting braces
                            brace_count = 0
                            json_end = json_start
                            for i in range(json_start, len(content)):
                                if content[i] == '{':
                                    brace_count += 1
                                elif content[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = i + 1
                                        break

                            if brace_count == 0 and json_end > json_start:
                                json_content = content[json_start:json_end]
                                parsed_result = json.loads(json_content)

                            # Check if this is the new grouped format (contains error type keys)
                            preset_error_types = ['Deletion', 'Insertion', 'Transposition', 'Scramble', 'Substitution', 'Homophone', 'Suffix', 'Hard Word', 'Extra Space', 'Extra Hyphen']
                            if any(error_type in parsed_result for error_type in preset_error_types):
                                # Extract user and review info from custom_id
                                custom_id = result['custom_id']
                                # Handle different custom_id formats:
                                # Format 1: {user_id}_review_{idx} (e.g., user_A_review_5) - len=4
                                # Format 2: {user_id}_review_{local_idx}_global_{global_idx} (e.g., user_A_review_5_global_15) - len=6
                                # Format 3: {user_id}_review_{local_idx}_global_{global_idx} (e.g., A13OFOB1394G31_review_4_global_4) - len=5
                                parts = custom_id.split('_')

                                if len(parts) == 4:  # Format 1: user_A_review_5
                                    user_id = parts[0] + '_' + parts[1]  # user_A
                                    local_review_idx = int(parts[3])    # 5
                                    global_review_idx = 0  # Not available in this format
                                    review_id = f"review_{local_review_idx}"
                                elif len(parts) == 5:  # Format 3: A13OFOB1394G31_review_4_global_4
                                    user_id = parts[0]  # A13OFOB1394G31
                                    local_review_idx = int(parts[2])    # 4
                                    global_review_idx = int(parts[4])   # 4
                                    review_id = f"review_{local_review_idx}"
                                elif len(parts) == 6:  # Format 2: user_A_review_5_global_15
                                user_id = parts[0] + '_' + parts[1]  # user_A
                                local_review_idx = int(parts[3])    # 5
                                global_review_idx = int(parts[5])   # 15
                                    review_id = f"review_{local_review_idx}"
                                else:
                                    raise ValueError(f"Unexpected custom_id format: {custom_id}")

                                # Process each error type group
                                for error_type in preset_error_types:
                                    if error_type in parsed_result and parsed_result[error_type]:
                                        for error in parsed_result[error_type]:
                                            # Get the review directly using the review index from custom_id
                                            review_text = sentences[global_review_idx] if sentences and global_review_idx < len(sentences) else ""

                                            # Filter out likely brand name misjudgments
                                            if _is_likely_brand_name_error(error, review_text):
                                                print(f"🚫 Filtered out likely brand name: '{error.get('original_word', '')}' in review: {review_text[:50]}...", flush=True)
                                                continue

                                            # Add review metadata and error type
                                            error['review_idx'] = global_review_idx
                                            error['review_id'] = review_id  # Review ID based on local index
                                            error['custom_id'] = custom_id
                                            error['error_type'] = error_type
                                            error['review'] = review_text  # Add the full review text
                                            all_errors.append(error)

                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse result for {result['custom_id']}: {e}", flush=True)
                        continue

    print(f"✅ Processed {len(all_errors)} spelling errors from batch results", flush=True)
    return all_errors

def _is_likely_brand_name_error(error: Dict[str, Any], review_text: str) -> bool:
    """
    Check if an error is likely a brand name misjudgment.

    EXTENDED Criteria:
    1. 首字母大写且不是句首，且修正方式仅为变成了普通字典词 (大小写变化)
    2. 或者reason中明确提到"brand"或"product name" (LLM已识别为品牌)

    Args:
        error: Error dictionary with 'original_word' and 'corrected_word'
        review_text: Full review text

    Returns:
        True if likely brand name error, False otherwise
    """
    original_word = error.get('original_word', '')
    corrected_word = error.get('corrected_word', '')
    reason = error.get('reason', '').lower()

    # Skip if either word is empty or too short
    if not original_word or not corrected_word or len(original_word) < 3:
        return False

    # Rule 1: 大小写变化过滤 (原规则)
    # 1. 首字母大写 (Original word starts with capital letter)
    if original_word[0].isupper():
        # 2. 修正方式仅为变成了普通字典词 (Correction is simply capitalization change)
        if original_word.lower() == corrected_word.lower():
            # 3. 不是句首 (NOT at sentence start)
            import re

            # Find all occurrences of the original word in the sentence
            pattern = r'\b' + re.escape(original_word) + r'\b'
            matches = list(re.finditer(pattern, review_text))

            # Also check compound words (e.g., "Crop" in "Crop-A-Dile")
            compound_matches = []
            if '-' in review_text:
                for word in review_text.split():
                    if '-' in word and original_word in word:
                        pos = review_text.find(word)
                        if pos >= 0:
                            compound_matches.append(pos)

            # Combine all positions
            all_positions = [match.start() for match in matches] + compound_matches

            # If word appears in sentence and not at start
            if all_positions:
                for pos in all_positions:
                    if pos > 8:  # Not within first 8 characters
                        return True

    # Rule 2: 品牌关键词过滤 (新规则)
    # 如果reason中明确提到brand或product name，过滤掉
    if 'brand' in reason or 'product name' in reason:
        return True

    return False

def load_user_data(json_file: str, num_users: int = 3) -> list:
    """
    Load user data and select specified number of users with more than 10 reviews for analysis.

    Args:
        json_file: Path to user reviews JSON file
        num_users: Number of users to select (default: 3)

    Returns:
        List of tuples: [(user_id1, review_texts1, review_count1), ...]
    """
    print(f"📖 Loading user data from: {json_file}", flush=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find users with more than 10 reviews
    qualified_users = [(user_id, user_data['review_count']) for user_id, user_data in data.items()
                      if user_data['review_count'] > 10]

    if len(qualified_users) < num_users:
        raise ValueError(f"Found only {len(qualified_users)} users with more than 10 reviews, need at least {num_users}")

    # Sort by review count (descending) and select top users
    qualified_users.sort(key=lambda x: x[1], reverse=True)
    selected_users = qualified_users[:num_users]

    print(f"👥 Found {len(qualified_users)} qualified users with >10 reviews", flush=True)

    result = []
    for i, (user_id, review_count) in enumerate(selected_users, 1):
        print(f"👤 Selected user {i}: {user_id} with {review_count} reviews", flush=True)

        # Extract review texts
        user_data = data[user_id]
        review_texts = []

        for review in user_data['reviews']:
            text = review.get('review_text', '').strip()
            if text:
                review_texts.append(text)

        print(f"✅ User {i}: Extracted {len(review_texts)} review texts", flush=True)
        result.append((user_id, review_texts, review_count))

    return result

def split_reviews_into_sentences(reviews: List[str]) -> List[str]:
    """
    Split reviews into individual sentences.

    Args:
        reviews: List of review texts

    Returns:
        List of individual sentences
    """
    import nltk
    from nltk.tokenize import sent_tokenize

    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    print(f"📝 Splitting {len(reviews)} reviews into sentences...", flush=True)

    all_sentences = []
    for review in reviews:
        # Keep original text intact for proper spelling analysis
        text = review
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)

    print(f"✅ Split into {len(all_sentences)} sentences", flush=True)
    return all_sentences

def main(num_users: int = 3, reviews_per_user: int = 10):
    """Main function for batch spelling error analysis of multiple users."""
    start_time = time.time()
    print(f"🚀 Starting Batch Spelling Error Analysis for {num_users} Users", flush=True)
    print(f"📍 Using model: {MODEL_NAME}", flush=True)
    print(f"📁 Input file: {INPUT_FILE}", flush=True)

    try:
        # Step 1: Load and prepare data for users
        print("\n📊 Step 1: Loading user data...", flush=True)
        users_data = load_user_data(INPUT_FILE, num_users)

        # Combine reviews from users
        all_reviews = []
        user_mapping = []  # Track which review belongs to which user

        for user_idx, (user_id, review_texts, total_reviews) in enumerate(users_data, 1):
            print(f"👤 User {user_idx}: {user_id} with {total_reviews} reviews", flush=True)

            # Take specified number of reviews from each user
            reviews_to_process = review_texts[:reviews_per_user]
            print(f"  📝 Taking {len(reviews_to_process)} reviews from {user_id}", flush=True)

            # Add reviews to combined list
            for review_idx, review in enumerate(reviews_to_process):
                all_reviews.append(review)
                user_mapping.append((user_id, review_idx))  # (user_id, local_review_index)

        print(f"\n✅ Combined {len(all_reviews)} reviews from {len(users_data)} users for batch processing", flush=True)

        # Step 2: Use reviews directly (no sentence splitting)
        print("\n📝 Step 2: Using reviews directly (no sentence splitting)...", flush=True)
        sentences = all_reviews  # Use all combined reviews

        # Step 3: Prepare batch requests for all reviews
        print("\n🔧 Step 3: Preparing batch requests for all reviews...", flush=True)
        batch_requests = prepare_batch_requests_combined(sentences, user_mapping)

        # Step 4: Save batch file
        timestamp = int(time.time())
        batch_filename = f"{BATCH_OUTPUT_DIR}/batch_spelling_analysis_combined_{timestamp}.jsonl"
        print("\n💾 Step 4: Saving batch file...", flush=True)
        save_batch_file(batch_requests, batch_filename)

        # Step 5: Upload batch file
        print("\n📤 Step 5: Uploading batch file...", flush=True)
        file_id = upload_batch_file(batch_filename)

        if not file_id:
            print("❌ Failed to upload batch file. Aborting.", flush=True)
            return

        # Step 6: Create batch job
        print("\n🚀 Step 6: Creating batch job...", flush=True)
        batch_name = f"spelling_analysis_combined_{len(users_data)}users_{len(all_reviews)}reviews"
        batch_id = create_batch_job(file_id, batch_name)

        if not batch_id:
            print("❌ Failed to create batch job. Aborting.", flush=True)
            return

        # Step 7: Monitor batch job
        print("\n👀 Step 7: Monitoring batch job...", flush=True)
        print("⚠️  This may take up to 24 hours. You can safely stop this script and check results later.", flush=True)
        batch_info = monitor_batch_job(batch_id)

        # Step 8: Download results
        if batch_info['status'] == 'completed':
            print("\n📥 Step 8: Downloading results...", flush=True)
            results_file = f"{BATCH_OUTPUT_DIR}/results_{batch_id}.jsonl"
            download_batch_results(batch_info, results_file)

            # Step 9: Process results
            print("\n🔍 Step 9: Processing results...", flush=True)
            spelling_errors = process_batch_results(results_file, sentences)

            # Step 10: Save final results
            print("\n💾 Step 10: Saving final results...", flush=True)

            # Calculate total word count
            total_words = sum(len(review.split()) for review in sentences)

            # Step 10: Group and save results by user
            print("\n💾 Step 10: Grouping results by user and saving...", flush=True)

            # Group errors by user
            user_errors = {}
            for user_id, _, _ in users_data:
                user_errors[user_id] = []

            for error in spelling_errors:
                    custom_id = error.get('custom_id', '')
                    if custom_id:
                        # Extract user_id from custom_id (format: user_A_review_localIdx_global_globalIdx)
                        parts = custom_id.split('_')
                    if len(parts) >= 3:
                        if len(parts) == 4:  # user_A_review_5
                            user_id = parts[0] + '_' + parts[1]
                            local_review_idx = int(parts[3])
                            review_id = f"review_{local_review_idx}"
                        elif len(parts) == 5:  # A13OFOB1394G31_review_4_global_4
                            user_id = parts[0]
                            local_review_idx = int(parts[2])
                            review_id = f"review_{local_review_idx}"
                        elif len(parts) == 6:  # user_A_review_5_global_15
                            user_id = parts[0] + '_' + parts[1]
                            local_review_idx = int(parts[3])
                            review_id = f"review_{local_review_idx}"
                        else:
                            continue  # Skip invalid format

                        if user_id in user_errors:
                            # Add review_id to error if not present
                            if 'review_id' not in error:
                                error['review_id'] = review_id
                            user_errors[user_id].append(error)

            # Initialize combined results structure
            combined_results = {
                'batch_id': batch_id,
                'model_used': MODEL_NAME,
                'total_users': len(users_data),
                'processing_time_hours': (time.time() - start_time) / 3600,
                'users': []
            }

            # Process each user separately
                            for user_idx, (user_id, review_texts, total_reviews) in enumerate(users_data, 1):
                                user_spelling_errors = user_errors[user_id]
                reviews_to_process = review_texts[:reviews_per_user]  # Same limit as used earlier

                    # Calculate word count for this user's reviews
                    total_words = sum(len(review.split()) for review in reviews_to_process)

                    # Define the 10 preset error types
                    preset_error_types = {
                        'Deletion': '机械性错误 (Mechanical/Typo) - 漏输 (Deletion)',
                        'Insertion': '机械性错误 (Mechanical/Typo) - 多输 (Insertion)',
                        'Transposition': '机械性错误 (Mechanical/Typo) - 字母换位 (Transposition)',
                        'Scramble': '机械性错误 (Mechanical/Typo) - 复杂混淆 (Scramble)',
                        'Substitution': '机械性错误 (Mechanical/Typo) - 字母替换 (Substitution)',
                        'Homophone': '语音/听觉错误 (Phonetic) - 同音/形近词 (Homophone)',
                        'Suffix': '语音/听觉错误 (Phonetic) - 后缀混淆 (Suffix)',
                        'Hard Word': '正字法/认知错误 (Orthographic) - 复杂词汇 (Hard Word)',
                        'Extra Space': '分隔错误 (Separation) - 多余空格 (Extra Space)',
                    'Extra Hyphen': '分隔错误 (Separation) - 多余连字符 (Extra Hyphen)'
                    }

                    # Initialize error groups for all 10 preset types
                    errors_by_type = {error_type: [] for error_type in preset_error_types.keys()}

                    # Group errors by preset error types with enhanced filtering
                    # Brand whitelist for arts and crafts domain
                    brand_whitelist = {
                        # Major brands
                        "Cricut", "Silhouette", "Sizzix", "Spellbinders", "Jolee's", "Jolee", "Martha Stewart", "Fiskars", "Olfa", "X-ACTO",
                        # Craft stores
                        "Michaels", "Joann", "Hobby Lobby", "AC Moore", "Ben Franklin",
                        # Product lines
                        "Nestabilities", "Spellbinders Nestabilities", "Embossabilities", "Cuttlebug", "Big Shot", "Wonder Cutter"
                    }

                    # Process each error for this user
                    for error in user_spelling_errors:
                        # Extract basic information
                        original_word = error['original_word'].strip("'\"")
                        corrected_word = error['corrected_word'].strip("'\"")
                        reason = error.get('reason', '')
                        reason_lower = reason.lower()

                        # Skip empty entries
                        if not original_word or not corrected_word:
                            continue  # Skip empty entries

                        # Skip "No error" type responses
                        if 'no error' in reason_lower or 'no spelling error' in reason_lower or 'correct' in reason_lower:
                            continue  # Skip this entry as it's not a real error

                        # Skip brand names that are in whitelist
                        if original_word in brand_whitelist or corrected_word in brand_whitelist:
                            continue  # Skip brand name corrections

                        # Skip grammar/morphology issues (not true spelling errors)
                        grammar_indicators = [
                            'subject-verb agreement', 'singular/plural', 'pluralization', 'agreement',
                            'word form', 'morphology', 'grammatical', 'tense', 'conjugation',
                            'incorrect pluralization', 'word form confusion'
                        ]
                        if any(grammar_issue in reason_lower for grammar_issue in grammar_indicators):
                            continue  # Skip grammar issues

                        # Skip if original and corrected words are essentially the same (no real change)
                        if original_word.lower() == corrected_word.lower():
                            continue  # Skip no-op corrections

                        # Get the error type that LLM has already assigned
                        error_type = error.get('error_type', 'Deletion')  # Default fallback

                        # Additional validation: ensure this is actually a spelling error
                        if not _is_genuine_spelling_error(original_word, corrected_word, reason_lower):
                            continue  # Skip non-spelling errors

                        # Create simplified error entry
                        simplified_error = {
                            'original_word': error['original_word'],
                            'corrected_word': error['corrected_word'],
                            'reason': error['reason'],
                            'review': error.get('review', ''),  # Add the original review text
                        'review_id': f"{user_id}_review_{error.get('review_idx', 0)}"  # Unique review identifier
                        }

                        # Add to the classified error type
                        errors_by_type[error_type].append(simplified_error)

                    # Calculate error type statistics
                    error_type_counts = {}
                    total_classified_errors = 0
                    spelling_errors_only = 0  # Exclude Extra Space and Extra Hyphen for error rate calculation

                    for error_type, errors in errors_by_type.items():
                        error_type_counts[error_type] = len(errors)
                        total_classified_errors += len(errors)
                        # Exclude Extra Space and Extra Hyphen from spelling error rate calculation
                        if error_type not in ['Extra Space', 'Extra Hyphen']:
                            spelling_errors_only += len(errors)

                    # Calculate error rate (errors per 100 words) using only spelling errors (excluding Extra Space/Hyphen)
                    error_rate_per_100_words = (spelling_errors_only / total_words * 100) if total_words > 0 else 0

                user_result = {
                        'user_id': user_id,
                        'total_reviews': total_reviews,
                        'reviews_processed': len(reviews_to_process),
                        'total_words': total_words,
                        'error_count': total_classified_errors,  # Total of all error types
                        'error_rate_per_100_words': round(error_rate_per_100_words, 2),  # Spelling errors only (excludes Extra Space/Hyphen)
                        'error_type_counts': error_type_counts,
                        'error_types': errors_by_type
                    }

                # Add user result to combined results
                combined_results['users'].append(user_result)

                print(f"✅ User {user_idx}/{len(users_data)} processed: {user_id}", flush=True)
                    print(f"   🎉 Found {total_classified_errors} spelling errors across {len(reviews_to_process)} reviews", flush=True)

            # Save combined results to single file
            combined_output_file = f"{BATCH_OUTPUT_DIR}/spelling_analysis_combined_{batch_id}.json"
            with open(combined_output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Combined results saved to: {combined_output_file}", flush=True)
            print(f"📊 Total users processed: {len(users_data)}", flush=True)
            else:
                print(f"❌ Batch job failed with status: {batch_info['status']}")
                return None

            # Continue with processing if batch succeeded
            preset_error_types = {
                'Deletion': '机械性错误 (Mechanical/Typo) - 漏输 (Deletion)',
                'Insertion': '机械性错误 (Mechanical/Typo) - 多输 (Insertion)',
                'Transposition': '机械性错误 (Mechanical/Typo) - 字母换位 (Transposition)',
                'Scramble': '机械性错误 (Mechanical/Typo) - 复杂混淆 (Scramble)',
                'Substitution': '机械性错误 (Mechanical/Typo) - 字母替换 (Substitution)',
                'Homophone': '语音/听觉错误 (Phonetic) - 同音/形近词 (Homophone)',
                'Suffix': '语音/听觉错误 (Phonetic) - 后缀混淆 (Suffix)',
                'Hard Word': '正字法/认知错误 (Orthographic) - 复杂词汇 (Hard Word)',
                'Extra Space': '分隔错误 (Separation) - 多余空格 (Extra Space)',
                'Extra Hyphen': '分隔错误 (Separation) - 多余连字符 (Extra Hyphen)'
            }

            # Initialize error groups for all 10 preset types
            errors_by_type = {error_type: [] for error_type in preset_error_types.keys()}

            # Group errors by preset error types with enhanced filtering
            # Brand whitelist for arts and crafts domain
            brand_whitelist = {
                # Major brands
                "Cricut", "Silhouette", "Sizzix", "Spellbinders", "Jolee's", "Jolee", "Martha Stewart", "Fiskars", "Olfa", "X-ACTO",
                # Craft stores
                "Michaels", "Joann", "Hobby Lobby", "AC Moore", "Ben Franklin",
                # Product lines
                "Nestabilities", "Spellbinders Nestabilities", "Embossabilities", "Cuttlebug", "Big Shot", "Wonder Cutter"
            }

            for error in spelling_errors:
                # Skip entries that are not actual errors (no error detected)
                reason_lower = error['reason'].lower()
                no_error_indicators = [
                    'no error', 'no error detected', 'no spelling error detected',
                    'no error found', 'correct spelling', 'no mistakes detected',
                    'no deletion error found', 'no insertion error found', 'no transposition error found'
                ]
                if any(no_error_phrase in reason_lower for no_error_phrase in no_error_indicators):
                    continue  # Skip this entry as it's not a real error

                # Skip brand names that are in whitelist
                original_word = error['original_word'].strip("'\"")
                corrected_word = error['corrected_word'].strip("'\"")
                if original_word in brand_whitelist or corrected_word in brand_whitelist:
                    continue  # Skip brand name corrections

                # Skip grammar/morphology issues (not true spelling errors)
                grammar_indicators = [
                    'subject-verb agreement', 'singular/plural', 'pluralization', 'agreement',
                    'word form', 'morphology', 'grammatical', 'tense', 'conjugation',
                    'incorrect pluralization', 'word form confusion'
                ]
                if any(grammar_issue in reason_lower for grammar_issue in grammar_indicators):
                    continue  # Skip grammar issues

                # Skip if original and corrected words are essentially the same (no real change)
                if original_word.lower() == corrected_word.lower():
                    continue  # Skip no-op corrections

                # Get the error type that LLM has already assigned
                error_type = error.get('error_type', 'Deletion')  # Default fallback

                # Additional validation: ensure this is actually a spelling error
                if not _is_genuine_spelling_error(original_word, corrected_word, reason_lower):
                    continue  # Skip non-spelling errors

                # Create simplified error entry
                simplified_error = {
                    'original_word': error['original_word'],
                    'corrected_word': error['corrected_word'],
                    'reason': error['reason'],
                    'review': error.get('review', ''),  # Add the original review text
                    'review_id': f"review_{error.get('review_idx', 0)}"  # Add review ID from error metadata
                }

                # Add to the classified error type
                errors_by_type[error_type].append(simplified_error)

            # Calculate error type statistics
            error_type_counts = {}
            total_classified_errors = 0
            spelling_errors_only = 0  # Exclude Extra Space and Extra Hyphen for error rate calculation

            for error_type, errors in errors_by_type.items():
                error_type_counts[error_type] = len(errors)
                total_classified_errors += len(errors)
                # Exclude Extra Space and Extra Hyphen from spelling error rate calculation
                if error_type not in ['Extra Space', 'Extra Hyphen']:
                    spelling_errors_only += len(errors)

            # Calculate error rate (errors per 100 words) using only spelling errors (excluding Extra Space/Hyphen)
            error_rate_per_100_words = (spelling_errors_only / total_words * 100) if total_words > 0 else 0

            final_result = {
                'user_id': user_id,
                'total_reviews': total_reviews,
                'reviews_processed': len(reviews_to_process),
                'total_words': total_words,
                'batch_id': batch_id,
                'model_used': MODEL_NAME,
                'error_count': total_classified_errors,  # Total of all error types
                'error_rate_per_100_words': round(error_rate_per_100_words, 2),  # Spelling errors only (excludes Extra Space/Hyphen)
                'error_type_counts': error_type_counts,
                'processing_time_hours': (time.time() - start_time) / 3600,
                'error_types': errors_by_type
            }

            output_file = f"{BATCH_OUTPUT_DIR}/spelling_analysis_{user_id}_{batch_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            print(f"✅ Final results saved to: {output_file}", flush=True)
            print(f"🎉 Found {total_classified_errors} spelling errors across {len(sentences)} reviews", flush=True)

            # Print final results content
            print("\n📄 Final Results Summary:")
            print("=" * 50)
            print(json.dumps(final_result, ensure_ascii=False, indent=2))
            print("=" * 50)

    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()

    print(".2f")
    return None

def test_batch_result_processing(full_process: bool = False):
    """
    Test script to process batch results and verify error parsing.

    Args:
        full_process: If True, process all responses instead of just first 100
    """
    import json
    import time

    results_file = "/home/wlia0047/ar57_scratch/wenyu/batch_results/results_batch_phmocsjztq.jsonl"
    print(f"🧪 Testing batch result processing: {results_file}", flush=True)

    # Load user data (simulate the 50 users selection)
    json_file = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json"
    print(f"📖 Loading user data from: {json_file}", flush=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find users with more than 10 reviews and select top 50
    qualified_users = [(user_id, user_data['review_count']) for user_id, user_data in data.items()
                      if user_data['review_count'] > 10]
    qualified_users.sort(key=lambda x: x[1], reverse=True)
    selected_users = qualified_users[:50]

    # Initialize user errors dictionary
    user_errors = {}
    for user_id, _ in selected_users:
        user_errors[user_id] = []

    print(f"👥 Selected {len(selected_users)} users for testing", flush=True)

    # Process batch results
    all_errors = []
    processed_count = 0
    error_count = 0
    start_time = time.time()

    preset_error_types = ['Deletion', 'Insertion', 'Transposition', 'Scramble', 'Substitution', 'Homophone', 'Suffix', 'Hard Word', 'Extra Space', 'Extra Hyphen']

    # Brand whitelist for filtering
    brand_whitelist = {
        "Cricut", "Silhouette", "Sizzix", "Spellbinders", "Jolee's", "Jolee", "Martha Stewart", "Fiskars", "Olfa", "X-ACTO",
        "Michaels", "Joann", "Hobby Lobby", "AC Moore", "Ben Franklin",
        "Nestabilities", "Spellbinders Nestabilities", "Embossabilities", "Cuttlebug", "Big Shot", "Wonder Cutter"
    }

    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not full_process and line_num > 100:  # Test with first 100 responses only unless full_process is True
                break

            result = json.loads(line.strip())
            processed_count += 1

            if 'response' in result and 'body' in result['response']:
                response_body = result['response']['body']

                if 'choices' in response_body and response_body['choices']:
                    content = response_body['choices'][0]['message']['content']
                    custom_id = result['custom_id']

                    # Parse JSON response
                    try:
                        json_start = content.find('{')
                        if json_start != -1:
                            brace_count = 0
                            json_end = json_start
                            for i in range(json_start, len(content)):
                                if content[i] == '{':
                                    brace_count += 1
                                elif content[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = i + 1
                                        break

                            if brace_count == 0 and json_end > json_start:
                                json_content = content[json_start:json_end]
                                parsed_result = json.loads(json_content)

                                # Check if this is the expected format
                                if any(error_type in parsed_result for error_type in preset_error_types):
                                    # Extract user and review info from custom_id
                                    parts = custom_id.split('_')
                                    if len(parts) >= 3:
                                        if len(parts) == 4:  # user_A_review_5
                                            user_id = parts[0] + '_' + parts[1]
                                            local_review_idx = int(parts[3])
                                            review_id = f"review_{local_review_idx}"
                                        elif len(parts) == 5:  # A13OFOB1394G31_review_4_global_4
                                            user_id = parts[0]
                                            local_review_idx = int(parts[2])
                                            review_id = f"review_{local_review_idx}"
                                        elif len(parts) == 6:  # user_A_review_5_global_15
                                            user_id = parts[0] + '_' + parts[1]
                                            local_review_idx = int(parts[3])
                                            review_id = f"review_{local_review_idx}"

                                    # Add errors to the appropriate user
                                        for error_type in preset_error_types:
                                            if error_type in parsed_result and parsed_result[error_type]:
                                                for error in parsed_result[error_type]:
                                                    original_word = error.get('original_word', '').strip("'\"")
                                                    corrected_word = error.get('corrected_word', '').strip("'\"")
                                                    reason = error.get('reason', '')

                                                    # Apply filters
                                                    if not original_word or not corrected_word:
                                                        continue  # Skip empty entries

                                                    if 'no error' in reason.lower() or 'no spelling error' in reason.lower() or 'correct' in reason.lower():
                                                        continue  # Skip "no error" responses

                                                    if original_word in brand_whitelist or corrected_word in brand_whitelist:
                                                        continue  # Skip brand names

                                                    # Grammar filters
                                                    grammar_indicators = [
                                                        'subject-verb agreement', 'singular/plural', 'pluralization', 'agreement',
                                                        'word form', 'morphology', 'grammatical', 'tense', 'conjugation',
                                                        'incorrect pluralization', 'word form confusion'
                                                    ]
                                                    if any(grammar_issue in reason.lower() for grammar_issue in grammar_indicators):
                                                        continue  # Skip grammar issues

                                                    if original_word.lower() == corrected_word.lower():
                                                        continue  # Skip no-op corrections

                                                    if not _is_genuine_spelling_error(original_word, corrected_word, reason):
                                                        continue  # Skip non-genuine errors

                                                    # Add the error
                                                    error['error_type'] = error_type
                                                    error['custom_id'] = custom_id
                                                    error['review_id'] = review_id  # Review ID based on local index
                                                    error['review_idx'] = 0  # Placeholder
                                                    error['review'] = ''  # Placeholder

                                                    if user_id in user_errors:
                                                        user_errors[user_id].append(error)
                                                        error_count += 1
                                                        if not full_process:  # Only print in test mode
                                                            print(f"✅ Found error for user {user_id}: {original_word} -> {corrected_word}")
                                                    else:
                                                        if not full_process:
                                                            print(f"⚠️ User {user_id} not in selected users list")

                    except json.JSONDecodeError as e:
                        if not full_process:
                            print(f"⚠️ Failed to parse JSON for {custom_id}: {e}")
                        continue

    # Generate summary
    processing_time = time.time() - start_time
    print(f"\n📊 Test Results Summary:")
    print(f"   📝 Processed {processed_count} responses in {processing_time:.2f}s")
    print(f"   🎯 Found {error_count} total spelling errors")

    user_stats = {}
    for user_id, errors in user_errors.items():
        if errors:
            user_stats[user_id] = len(errors)

    if user_stats:
        print(f"   👥 Errors found for {len(user_stats)} users:")
        for user_id, count in sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
            print(f"      {user_id}: {count} errors")

        # Generate combined results
        combined_results = {
            'batch_id': 'test_batch_phmocsjztq',
            'model_used': 'Qwen/QwQ-32B',
            'total_users': len(selected_users),
            'processing_time_hours': processing_time / 3600,
            'users': []
        }

        for user_id, review_count in selected_users:
            user_result = {
                'user_id': user_id,
                'total_reviews': review_count,
                'reviews_processed': 30,  # Placeholder
                'total_words': 0,  # Placeholder
                'error_count': len(user_errors[user_id]),
                'error_rate_per_100_words': 0.0,  # Placeholder
                'error_type_counts': {},
                'error_types': {}
            }

            # Count error types
            for error in user_errors[user_id]:
                error_type = error['error_type']
                if error_type not in user_result['error_type_counts']:
                    user_result['error_type_counts'][error_type] = 0
                user_result['error_type_counts'][error_type] += 1

                if error_type not in user_result['error_types']:
                    user_result['error_types'][error_type] = []
                user_result['error_types'][error_type].append({
                    'original_word': error['original_word'],
                    'corrected_word': error['corrected_word'],
                    'reason': error['reason'],
                    'review_id': error.get('review_id', ''),
                    'custom_id': error.get('custom_id', ''),
                    'review': error.get('review', '')
                })

            combined_results['users'].append(user_result)

        # Save results
        output_file = "/home/wlia0047/ar57_scratch/wenyu/batch_results/test_spelling_analysis_combined_batch_phmocsjztq.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Test results saved to: {output_file}")
    else:
        print("   ❌ No errors found for any user")

    return error_count > 0

if __name__ == "__main__":
    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        full_process = len(sys.argv) > 2 and sys.argv[2] == "full"
        success = test_batch_result_processing(full_process)
        if success:
            print("\n✅ Test passed: Errors were correctly parsed and grouped!")
        else:
            print("\n❌ Test failed: No errors were found")
        sys.exit(0)

    # Parse command line arguments
    num_users = 3  # Default value
    reviews_per_user = 10  # Default value

    if len(sys.argv) > 1:
        try:
            num_users = int(sys.argv[1])
            if num_users < 1:
                raise ValueError("Number of users must be at least 1")
        except ValueError as e:
            print(f"❌ Error: Invalid number of users. Please provide a positive integer. {e}", flush=True)
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            reviews_per_user = int(sys.argv[2])
            if reviews_per_user < 1:
                raise ValueError("Reviews per user must be at least 1")
        except ValueError as e:
            print(f"❌ Error: Invalid reviews per user. Please provide a positive integer. {e}", flush=True)
            sys.exit(1)

    print(f"🔧 Configuration: {num_users} users, {reviews_per_user} reviews per user", flush=True)

    # Create output directory
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    main(num_users, reviews_per_user)