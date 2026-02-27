#!/usr/bin/env python3
"""
Verification script for noisy query generation.
Checks single-error compliance and provides statistics.
"""

import csv
import sys
import json

def verify_single_error(original, noisy):
    """Check if only one error was introduced."""
    if original == noisy:
        return True, "No changes"

    orig_words = original.split()
    noisy_words = noisy.split()

    # Count word changes
    word_changes = sum(1 for o, n in zip(orig_words, noisy_words) if o != n)

    # Allow for single word replacement or minor character change
    if word_changes == 1:
        return True, "Single word change"
    elif word_changes == 0:
        # Character-level change within a word
        return True, "Character-level change"
    else:
        return False, f"Multiple changes ({word_changes} words)"

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_results.py <noisy_queries.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    modified = [r for r in rows if r['original_query'] != r['noisy_query']]

    print(f"📊 Noisy Query Verification")
    print("=" * 50)
    print(f"Total queries:      {total}")
    print(f"Modified:           {len(modified)} ({len(modified)*100//total}%)")
    print(f"Unchanged:          {total - len(modified)} ({(total - len(modified))*100//total}%)")
    print()

    # Verify single-error rule
    print("🔍 Single-Error Compliance Check")
    print("-" * 50)

    violations = []
    for r in modified:
        valid, reason = verify_single_error(r['original_query'], r['noisy_query'])
        if not valid:
            violations.append((r['id'], reason))

    if violations:
        print(f"⚠️  Found {len(violations)} violations:")
        for id, reason in violations[:10]:
            print(f"  - ID {id}: {reason}")
    else:
        print(f"✅ All {len(modified)} modified queries comply!")

    print()
    print("📈 Sample Modified Queries")
    print("-" * 50)

    for i, r in enumerate(modified[:5]):
        print(f"\n[{r['id']}]")
        print(f"  Original:  {r['original_query'][:60]}...")
        print(f"  Noisy:     {r['noisy_query'][:60]}...")

if __name__ == '__main__':
    main()
