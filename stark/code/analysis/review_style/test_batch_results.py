#!/usr/bin/env python3
"""
Standalone test script for processing batch results.

This script can be used to test the batch result processing functionality
without running the full pipeline.

Usage:
    python test_batch_results.py [results_file] [num_users] [full]

Arguments:
    results_file: Path to batch results JSONL file (default: auto-detect)
    num_users: Number of users to process (default: 50)
    full: Process all responses instead of first 100 (default: False)
"""

import json
import time
import sys
import os

def _is_genuine_spelling_error(original: str, corrected: str, reason: str) -> bool:
    """Validate if this is a genuine spelling error."""
    if original == corrected:
        return False

    original_chars = original.lower()
    corrected_chars = corrected.lower()

    if abs(len(original_chars) - len(corrected_chars)) > 3:
        return False

    if original_chars + 's' == corrected_chars or corrected_chars + 's' == original_chars:
        return False

    verb_suffixes = ['s', 'ed', 'ing', 'er', 'est']
    for suffix in verb_suffixes:
        if original_chars + suffix == corrected_chars or corrected_chars + suffix == original_chars:
            return False

    return True

def test_batch_result_processing(results_file=None, num_users=50, full_process=False):
    """
    Test script to process batch results and verify error parsing.

    Args:
        results_file: Path to batch results file
        num_users: Number of users to select
        full_process: If True, process all responses
    """
    if not results_file:
        # Auto-detect the latest results file
        batch_dir = "/home/wlia0047/ar57_scratch/wenyu/batch_results"
        results_files = [f for f in os.listdir(batch_dir) if f.startswith('results_batch_') and f.endswith('.jsonl')]
        if results_files:
            # Sort by modification time, get the latest
            results_files.sort(key=lambda x: os.path.getmtime(os.path.join(batch_dir, x)), reverse=True)
            results_file = os.path.join(batch_dir, results_files[0])
        else:
            print("❌ No results files found in batch_results directory")
            return False

    print(f"🧪 Testing batch result processing: {results_file}", flush=True)

    # Load user data
    json_file = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json"
    print(f"📖 Loading user data from: {json_file}", flush=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        user_data_full = json.load(f)

    # Find users with more than 10 reviews and select top N
    qualified_users = [(user_id, user_data_full[user_id]['review_count']) for user_id in user_data_full.keys()
                      if user_data_full[user_id]['review_count'] > 10]
    qualified_users.sort(key=lambda x: x[1], reverse=True)
    selected_users = qualified_users[:num_users]

    # Create user to reviews mapping for lookup
    user_reviews_map = {}
    for user_id, _ in selected_users:
        if user_id in user_data_full and 'reviews' in user_data_full[user_id]:
            user_reviews_map[user_id] = [review.get('review_text', '') for review in user_data_full[user_id]['reviews']]

    # Initialize user errors dictionary
    user_errors = {}
    for user_id, _ in selected_users:
        user_errors[user_id] = []

    print(f"👥 Selected {len(selected_users)} users for testing", flush=True)

    # Function to get review text by custom_id
    def get_review_text(custom_id, user_reviews_map):
        """Get review text based on custom_id"""
        parts = custom_id.split('_')
        if len(parts) >= 3:
            if len(parts) == 4:  # user_A_review_5
                user_id = parts[0] + '_' + parts[1]
                local_idx = int(parts[3])
            elif len(parts) == 5:  # A13OFOB1394G31_review_4_global_4
                user_id = parts[0]
                local_idx = int(parts[2])
            else:  # user_A_review_5_global_15
                user_id = parts[0] + '_' + parts[1]
                local_idx = int(parts[3])

            if user_id in user_reviews_map and local_idx < len(user_reviews_map[user_id]):
                return user_reviews_map[user_id][local_idx]
        return f"Review text not found for {custom_id}"

    # Process batch results
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
                                    review_id = "review_0"  # Default
                                    if len(parts) >= 3:
                                        review_id = "review_0"  # Default
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
                                                    error['review_idx'] = 0  # Placeholder
                                                    error['review_id'] = review_id  # Review ID based on local index
                                                    error['review'] = get_review_text(custom_id, user_reviews_map)  # Get actual review text

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
            'batch_id': f'test_{os.path.basename(results_file).replace("results_", "").replace(".jsonl", "")}',
            'model_used': 'Qwen/QwQ-32B',
            'total_users': len(selected_users),
            'processing_time_hours': processing_time / 3600,
            'users': []
        }

        for user_id, review_count in selected_users:
            # Calculate actual word count for this user
            user_reviews = user_reviews_map.get(user_id, [])
            total_words = sum(len(review.split()) for review in user_reviews[:30])  # Use first 30 reviews like main script

            # Calculate error rate
            spelling_errors_only = 0
            for error in user_errors[user_id]:
                error_type = error['error_type']
                if error_type not in ['Extra Space', 'Extra Hyphen']:
                    spelling_errors_only += 1

            error_rate_per_100_words = (spelling_errors_only / total_words * 100) if total_words > 0 else 0.0

            user_result = {
                'user_id': user_id,
                'total_reviews': review_count,
                'reviews_processed': len(user_reviews[:30]),  # Actual count
                'total_words': total_words,
                'error_count': len(user_errors[user_id]),
                'error_rate_per_100_words': round(error_rate_per_100_words, 2),
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
        output_file = f"/home/wlia0047/ar57_scratch/wenyu/batch_results/test_spelling_analysis_{os.path.basename(results_file).replace('results_', '').replace('.jsonl', '')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Test results saved to: {output_file}")
    else:
        print("   ❌ No errors found for any user")

    return error_count > 0

if __name__ == "__main__":
    results_file = None
    num_users = 50
    full_process = False

    # Parse arguments
    args = sys.argv[1:]
    if args:
        if not args[0].startswith('-'):
            results_file = args[0]
        else:
            print("Usage: python test_batch_results.py [results_file] [num_users] [full]")
            sys.exit(1)

    if len(args) > 1:
        try:
            num_users = int(args[1])
        except ValueError:
            print(f"❌ Invalid num_users: {args[1]}")
            sys.exit(1)

    if len(args) > 2 and args[2].lower() == 'full':
        full_process = True

    success = test_batch_result_processing(results_file, num_users, full_process)
    if success:
        print("\n✅ Test passed: Errors were correctly parsed and grouped!")
    else:
        print("\n❌ Test failed: No errors were found")
        sys.exit(1)