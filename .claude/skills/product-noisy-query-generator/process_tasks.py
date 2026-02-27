#!/usr/bin/env python3
"""
Process noisy query tasks with ONE ERROR PER SENTENCE rule.
This script helps apply the highest-weighted trigger only.
"""

import json
import csv
import sys

def load_tasks(filepath):
    """Load tasks from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_significant_triggers(triggers, min_weight=0.08):
    """Filter triggers by minimum weight and sort by weight descending."""
    significant = [t for t in triggers if t.get('coef', 0) >= min_weight]
    return sorted(significant, key=lambda x: x.get('coef', 0), reverse=True)

def apply_single_error(original_query, triggers):
    """
    Apply ONLY the highest-weighted trigger to the query.
    Returns the modified query and info about what was applied.
    """
    significant = get_significant_triggers(triggers)

    if not significant:
        return original_query, None

    # Use only the HIGHEST weighted trigger
    trigger = significant[0]
    query = original_query

    if trigger['type'] == 'lexical' and 'value' in trigger:
        # Simple replacement
        target = trigger.get('target', '')
        value = trigger.get('value', '')
        if target and value:
            query = query.replace(target, value, 1)  # Replace only first occurrence

    elif trigger['type'] in ['structure_length', 'structure_speed']:
        # For structure triggers, apply common errors
        import re

        # Try to find and replace common words
        for word, replacement in [
            ("beautiful", "beatiful"),
            ("really", "realley"),
            ("separate", "seperate"),
            ("until", "untill"),
            ("beginning", "begining"),
            ("definitely", "definitly"),
            ("necessary", "neccesary"),
            ("received", "recieved"),
            ("colored", "coloured"),
            ("color", "colour"),
            ("favorite", "favourite"),
        ]:
            if word.lower() in query.lower():
                pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                query = pattern.sub(replacement, query, count=1)
                break
        else:
            # If no common word, try "that are/is" -> "that"
            if " that are " in query.lower():
                query = re.sub(r'\bthat are\b', 'that', query, count=1, flags=re.IGNORECASE)
            elif " that is " in query.lower():
                query = re.sub(r'\bthat is\b', 'that', query, count=1, flags=re.IGNORECASE)
            else:
                # Last resort: find a word with double letters and make it single, or vice versa
                words = query.split()
                for i, word in enumerate(words):
                    clean = re.sub(r'[^\w]', '', word.lower())
                    # Apply transposition to common words
                    if clean in ["will", "well", "look", "good", "see", "been", "keep"]:
                        if clean == "will":
                            words[i] = word.replace("will", "wii")
                            break
                        elif clean == "look":
                            words[i] = word.replace("look", "loook")
                            break
                    # Double a letter in words ending with consonant + 'l' or 't'
                    elif re.search(r'[aeiou][lt]$', clean):
                        # Double the last letter
                        if word[-1].isalpha():
                            words[i] = word[:-1] + word[-1] * 2
                            break
                query = ' '.join(words)

    elif trigger['type'] == 'relative_copula_drop':
        # "that are" -> "that", "that is" -> "that"
        import re
        if " that are " in query.lower():
            query = re.sub(r'\bthat are\b', 'that', query, count=1, flags=re.IGNORECASE)
        elif " that is " in query.lower():
            query = re.sub(r'\bthat is\b', 'that', query, count=1, flags=re.IGNORECASE)

    elif trigger['type'] == 'agreement_error':
        import re
        verb = trigger.get('target', '')
        if verb.lower() in ['are', 'do', 'have']:
            deg_map = {'are': 'is', 'do': 'does', 'have': 'has'}
            query = re.sub(r'\b' + verb + r'\b', deg_map[verb.lower()], query, count=1, flags=re.IGNORECASE)

    return query, trigger

def process_batch(tasks, start_id, end_id, output_csv):
    """Process a batch of tasks and append to CSV."""
    results = []

    for task in tasks:
        task_id = int(task['id'])
        if start_id <= task_id < end_id:
            original = task['original_query']
            triggers = task.get('triggers_detail', [])

            if task['risk_status'] == 'HIGH RISK':
                noisy, applied = apply_single_error(original, triggers)
                applied_info = f"{applied['type']}:{applied.get('target', 'N/A')}" if applied else "None"
            else:
                # LOW RISK - keep original
                noisy = original
                applied_info = "LOW_RISK_KEEP"

            results.append({
                'id': task['id'],
                'original_query': original,
                'noisy_query': noisy,
                'answer_ids_source': task['answer_ids_source'],
                'risk_status': task['risk_status'],
                'applied_trigger': applied_info
            })

    # Write to CSV
    with open(output_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([r['id'], r['original_query'], r['noisy_query'], r['answer_ids_source']])

    return results

def main():
    if len(sys.argv) < 4:
        print("Usage: process_tasks.py <tasks_json> <output_csv> <start_id> <end_id>")
        print("Example: process_tasks.py noisy_query_tasks.json noisy_queries.csv 0 20")
        sys.exit(1)

    tasks_file = sys.argv[1]
    output_csv = sys.argv[2]
    start_id = int(sys.argv[3])
    end_id = int(sys.argv[4])

    tasks = load_tasks(tasks_file)
    results = process_batch(tasks, start_id, end_id, output_csv)

    print(f"✅ Processed {len(results)} tasks (IDs {start_id}-{end_id-1})")
    print(f"📁 Appended to {output_csv}")
    print()

    # Show sample
    print("Sample results:")
    for r in results[:5]:
        changed = "✓" if r['original_query'] != r['noisy_query'] else "○"
        print(f"  [{r['id']}] {changed} {r['risk_status']} - {r['applied_trigger']}")
        if r['original_query'] != r['noisy_query']:
            print(f"      Original:  {r['original_query'][:60]}...")
            print(f"      Noisy:     {r['noisy_query'][:60]}...")

if __name__ == '__main__':
    main()
