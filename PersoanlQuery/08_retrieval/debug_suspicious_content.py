#!/usr/bin/env python3
"""
Debug script to check suspicious content in detail.
"""
import gzip
import json
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
clean_data = utils.clean_data


def load_samples_with_suspicious_content(meta_file: str, n_samples: int = 5000) -> list:
    """Load samples that might have suspicious content"""
    log_with_timestamp(f"Loading samples to check suspicious content...")
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

    return samples


def analyze_suspicious_patterns(samples: list) -> None:
    """Analyze what the suspicious patterns actually are"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("SUSPICIOUS PATTERN DETAILS")
    log_with_timestamp("="*60)

    suspicious_keywords = {
        'function': [],
        'script': [],
        'css': [],
        'javascript': [],
        'html': [],
        'style': [],
    }

    for asin, item in samples:
        desc = item.get('description', '')
        if isinstance(desc, list):
            desc = ' '.join(str(x) for x in desc if x)
        elif not isinstance(desc, str):
            desc = str(desc)

        desc_lower = desc.lower()

        for keyword in suspicious_keywords:
            if keyword in desc_lower:
                # Find context around the keyword
                idx = desc_lower.find(keyword)
                context = desc[max(0, idx-30):min(len(desc), idx+50)]
                suspicious_keywords[keyword].append((asin, context))

    for keyword, occurrences in suspicious_keywords.items():
        if occurrences:
            log_with_timestamp(f"\n--- Keyword '{keyword}' found in {len(occurrences)} documents ---")
            for asin, context in occurrences[:3]:
                log_with_timestamp(f"\n  ASIN: {asin}")
                log_with_timestamp(f"  Context: ...{context}...")


def check_real_issues(samples: list) -> None:
    """Check for REAL issues that might affect embedding quality"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("REAL CONTENT QUALITY ISSUES")
    log_with_timestamp("="*60)

    issues = {
        'very_short_title': [],      # Title < 10 chars
        'title_equals_description': [],  # Title and description are the same
        'gibberish_content': [],     # Random characters
        'all_caps_description': [],  # ALL CAPS description (annoying)
        'copy_paste_repeat': [],      # Same phrase repeated
    }

    for asin, item in samples:
        title = item.get('title', '')
        desc = item.get('description', '')
        if isinstance(desc, list):
            desc = ' '.join(str(x) for x in desc if x)
        elif not isinstance(desc, str):
            desc = str(desc)

        # Very short title
        if len(title) < 10:
            issues['very_short_title'].append((asin, title))

        # Title equals description (exact match)
        if title and desc and title.strip().lower() == desc.strip().lower()[:len(title)]:
            issues['title_equals_description'].append((asin, title[:100]))

        # ALL CAPS description (more than 50% uppercase)
        if desc and len(desc) > 20:
            upper_count = sum(1 for c in desc if c.isupper())
            if upper_count / len(desc) > 0.5:
                issues['all_caps_description'].append((asin, desc[:100]))

        # Gibberish check (too many unusual characters)
        unusual_chars = len(re.findall(r'[^\w\s\.\,\!\?\-\:\;\'\"]', desc))
        if unusual_chars / max(len(desc), 1) > 0.1:
            issues['gibberish_content'].append((asin, desc[:100]))

    for issue_name, examples in issues.items():
        log_with_timestamp(f"\n{issue_name}: {len(examples)} found")
        if examples:
            for asin, example in examples[:2]:
                log_with_timestamp(f"  ASIN {asin}: {repr(example)[:100]}")


def show_sample_built_docs(samples: list) -> None:
    """Show complete built documents for analysis"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("SAMPLE COMPLETE BUILT DOCUMENTS")
    log_with_timestamp("="*60)

    from utils import utils
    build_stark_document = utils.build_stark_document

    # Show 5 random samples
    import random
    random.seed(42)
    selected = random.sample(samples, min(5, len(samples)))

    for asin, item in selected:
        log_with_timestamp(f"\n{'='*60}")
        log_with_timestamp(f"ASIN: {asin}")
        log_with_timestamp(f"{'='*60}")

        # Raw fields
        log_with_timestamp("\nRAW FIELDS:")
        log_with_timestamp(f"  title: {repr(item.get('title', '')[:200])}")

        desc = item.get('description', '')
        if isinstance(desc, list):
            desc_str = ' '.join(str(x) for x in desc if x)
        else:
            desc_str = str(desc)
        log_with_timestamp(f"  description: {repr(desc_str[:300])}")

        features = item.get('features', [])
        if isinstance(features, list):
            feat_str = str(features[:3])
        else:
            feat_str = str(features)
        log_with_timestamp(f"  features: {feat_str[:200]}")

        # Built document
        doc = build_stark_document(item, None, add_rel=True, compact=False)
        log_with_timestamp(f"\nBUILT DOCUMENT ({len(doc)} chars):")
        log_with_timestamp(doc)


def main():
    meta_file = "/fs04/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz"

    log_with_timestamp("="*60)
    log_with_timestamp("SUSPICIOUS CONTENT DEBUG")
    log_with_timestamp("="*60)

    samples = load_samples_with_suspicious_content(meta_file, n_samples=5000)
    log_with_timestamp(f"Loaded {len(samples)} samples")

    analyze_suspicious_patterns(samples)
    check_real_issues(samples)
    show_sample_built_docs(samples)

    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("DEBUG COMPLETE")
    log_with_timestamp("="*60)


if __name__ == "__main__":
    main()
