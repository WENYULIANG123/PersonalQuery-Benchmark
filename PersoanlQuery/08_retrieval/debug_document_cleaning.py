#!/usr/bin/env python3
"""
Debug script to analyze document content and explore cleaning strategies.
"""
import gzip
import json
import sys
import os
from collections import Counter

# Add utils path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
build_stark_document = utils.build_stark_document
clean_data = utils.clean_data


def load_sample_metadata(meta_file: str, n_samples: int = 100) -> list:
    """Load sample metadata for analysis"""
    log_with_timestamp(f"Loading {n_samples} samples from {meta_file}...")
    samples = []
    open_func = gzip.open if meta_file.endswith('.gz') else open

    with open_func(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                item = json.loads(line.strip())
                asin = item.get('asin') or item.get('parent_asin')
                samples.append((asin, item))
            except:
                continue

    log_with_timestamp(f"Loaded {len(samples)} samples")
    return samples


def analyze_field_lengths(samples: list) -> None:
    """Analyze length distribution of key fields"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("FIELD LENGTH ANALYSIS")
    log_with_timestamp("="*60)

    for asin, item in samples[:5]:
        log_with_timestamp(f"\n--- Sample ASIN: {asin} ---")
        for field in ['title', 'description', 'features']:
            value = item.get(field, '')
            if isinstance(value, list):
                lengths = [len(str(v)) for v in value]
                log_with_timestamp(f"  {field}: LIST with {len(value)} items, total_chars={sum(lengths)}")
            elif isinstance(value, str):
                log_with_timestamp(f"  {field}: STRING with {len(value)} chars")
            else:
                log_with_timestamp(f"  {field}: {type(value).__name__}")


def analyze_raw_document(samples: list) -> None:
    """Analyze raw document text before/after cleaning"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("RAW DOCUMENT ANALYSIS")
    log_with_timestamp("="*60)

    for i, (asin, item) in enumerate(samples[:10]):
        doc = build_stark_document(item, None, add_rel=False, compact=False)
        tokens = doc.split()
        log_with_timestamp(f"\n--- Document {i+1}: ASIN={asin} ---")
        log_with_timestamp(f"  Total chars: {len(doc)}")
        log_with_timestamp(f"  Total tokens: {len(tokens)}")
        log_with_timestamp(f"  First 300 chars:\n{doc[:300]}...")
        log_with_timestamp(f"  Last 100 chars:\n...{doc[-100:]}")


def find_extreme_cases(samples: list, field: str = 'description') -> None:
    """Find documents with extremely long fields"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp(f"EXTREME CASES - Field: {field}")
    log_with_timestamp("="*60)

    extreme_cases = []
    for asin, item in samples:
        value = item.get(field, '')
        if isinstance(value, str) and len(value) > 5000:
            extreme_cases.append((asin, len(value), value[:500]))

    extreme_cases.sort(key=lambda x: x[1], reverse=True)

    log_with_timestamp(f"Found {len(extreme_cases)} documents with {field} > 5000 chars")
    for asin, length, preview in extreme_cases[:3]:
        log_with_timestamp(f"\n--- ASIN: {asin} (length={length}) ---")
        log_with_timestamp(f"Preview: {preview[:300]}...")


def suggest_cleaning_rules(samples: list) -> None:
    """Suggest cleaning rules based on analysis"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("SUGGESTED CLEANING RULES")
    log_with_timestamp("="*60)

    issues_found = {
        'html_tags': 0,
        'url_in_text': 0,
        'extreme_length': 0,
        'special_chars': 0,
        'empty_fields': 0,
    }

    for asin, item in samples:
        # Check for HTML
        desc = item.get('description', '')
        if isinstance(desc, str):
            if '<' in desc and '>' in desc:
                issues_found['html_tags'] += 1
            if 'http' in desc.lower():
                issues_found['url_in_text'] += 1

        # Check for extreme length
        if isinstance(desc, str) and len(desc) > 10000:
            issues_found['extreme_length'] += 1

        # Check for empty fields
        if not desc or (isinstance(desc, list) and len(desc) == 0):
            issues_found['empty_fields'] += 1

    log_with_timestamp("\nIssue counts in sample:")
    for issue, count in issues_found.items():
        log_with_timestamp(f"  {issue}: {count} ({count/len(samples)*100:.1f}%)")


def demo_cleaning() -> None:
    """Demo various cleaning approaches"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("CLEANING DEMO")
    log_with_timestamp("="*60)

    test_cases = [
        "This is a <b>normal</b> product description with HTML tags",
        "Check out https://example.com/product/123 for more info",
        "Product description with special chars: \x00\x01\x02 null bytes",
        "Very long " + "word " * 1000 + "description that needs truncation",
        "Multiple     spaces    and\ttabs\t\rare\twrong",
        None,  # None case
        "",    # Empty case
        ["item1", "item2", ""],  # List with empty
    ]

    for i, test in enumerate(test_cases):
        log_with_timestamp(f"\n--- Test case {i+1}: {repr(test)[:80]} ---")
        try:
            cleaned = clean_data(test)
            log_with_timestamp(f"  Cleaned: {repr(cleaned)[:80]}")
            log_with_timestamp(f"  Length: {len(cleaned) if cleaned else 0}")
        except Exception as e:
            log_with_timestamp(f"  ERROR: {e}")


def main():
    meta_file = "/fs04/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz"

    log_with_timestamp("="*60)
    log_with_timestamp("DOCUMENT CLEANING DEBUG SCRIPT")
    log_with_timestamp("="*60)
    log_with_timestamp(f"Meta file: {meta_file}")

    # Load samples
    samples = load_sample_metadata(meta_file, n_samples=1000)

    # Run analyses
    analyze_field_lengths(samples)
    analyze_raw_document(samples)
    find_extreme_cases(samples, 'description')
    find_extreme_cases(samples, 'features')
    suggest_cleaning_rules(samples)
    demo_cleaning()

    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("DEBUG COMPLETE")
    log_with_timestamp("="*60)


if __name__ == "__main__":
    main()
