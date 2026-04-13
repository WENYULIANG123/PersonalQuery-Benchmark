#!/usr/bin/env python3
"""
Debug script to analyze raw document content and cleaning effects.
"""
import gzip
import json
import sys
import os
import re
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
build_stark_document = utils.build_stark_document
clean_data = utils.clean_data


def load_raw_metadata(meta_file: str, n_samples: int = 100) -> list:
    """Load raw metadata without any cleaning"""
    log_with_timestamp(f"Loading {n_samples} raw samples from {meta_file}...")
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

    log_with_timestamp(f"Loaded {len(samples)} raw samples")
    return samples


def analyze_raw_content(samples: list) -> None:
    """Analyze raw (uncleaned) content of key fields"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("RAW CONTENT ANALYSIS (BEFORE CLEANING)")
    log_with_timestamp("="*60)

    issues = {
        'has_html_tag': 0,
        'has_url': 0,
        'has_control_char': 0,
        'has_unicode_escape': 0,
        'has_emoji': 0,
        'has_price_pattern': 0,
        'has_asin_pattern': 0,
        'extreme_length': 0,
        'empty': 0,
        'suspicious_content': 0,
    }

    for asin, item in samples:
        # Check description
        desc = item.get('description', '')
        if isinstance(desc, list):
            desc = ' '.join(str(x) for x in desc if x)
        elif not isinstance(desc, str):
            desc = str(desc)

        if not desc:
            issues['empty'] += 1
        else:
            if '<' in desc and '>' in desc:
                issues['has_html_tag'] += 1
            if 'http' in desc.lower():
                issues['has_url'] += 1
            if re.search(r'\\x[0-9a-fA-F]{2}', desc):
                issues['has_unicode_escape'] += 1
            if len(desc) > 5000:
                issues['extreme_length'] += 1
            # Check for control characters
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', desc):
                issues['has_control_char'] += 1
            # Check for emoji/non-ASCII
            if re.search(r'[\U0001F300-\U0001F9FF]', desc):
                issues['has_emoji'] += 1
            # Suspicious patterns
            if 'function' in desc.lower() or 'script' in desc.lower() or 'css' in desc.lower():
                issues['suspicious_content'] += 1

    log_with_timestamp("\nIssue counts in raw content:")
    for issue, count in issues.items():
        pct = count/len(samples)*100 if len(samples) > 0 else 0
        log_with_timestamp(f"  {issue}: {count} ({pct:.1f}%)")


def show_detailed_raw_samples(samples: list) -> None:
    """Show detailed raw content for selected samples"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("DETAILED RAW CONTENT SAMPLES")
    log_with_timestamp("="*60)

    # Find samples with different characteristics
    html_samples = []
    url_samples = []
    long_samples = []

    for asin, item in samples:
        desc = item.get('description', '')
        if isinstance(desc, list):
            desc_full = ' '.join(str(x) for x in desc if x)
        elif isinstance(desc, str):
            desc_full = desc
        else:
            desc_full = str(desc)

        if '<' in desc_full and '>' in desc_full:
            html_samples.append((asin, item, desc_full[:500]))
        if 'http' in desc_full.lower():
            url_samples.append((asin, item, desc_full[:500]))
        if isinstance(desc, str) and len(desc) > 2000:
            long_samples.append((asin, item, desc_full[:500]))

    # Show HTML samples
    if html_samples:
        log_with_timestamp(f"\n--- SAMPLES WITH HTML TAGS ({len(html_samples)} found) ---")
        for asin, item, preview in html_samples[:2]:
            log_with_timestamp(f"\nASIN: {asin}")
            log_with_timestamp(f"Raw description (first 500 chars):\n{preview}...")

    # Show URL samples
    if url_samples:
        log_with_timestamp(f"\n--- SAMPLES WITH URLs ({len(url_samples)} found) ---")
        for asin, item, preview in url_samples[:2]:
            log_with_timestamp(f"\nASIN: {asin}")
            log_with_timestamp(f"Raw description (first 500 chars):\n{preview}...")

    # Show long samples
    if long_samples:
        log_with_timestamp(f"\n--- LONG DESCRIPTION SAMPLES ({len(long_samples)} found) ---")
        for asin, item, preview in long_samples[:3]:
            log_with_timestamp(f"\nASIN: {asin}")
            log_with_timestamp(f"Length: {len(item.get('description', ''))} chars")
            log_with_timestamp(f"Raw description (first 500 chars):\n{preview}...")


def compare_before_after_cleaning(samples: list) -> None:
    """Compare content before and after cleaning"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("BEFORE VS AFTER CLEANING COMPARISON")
    log_with_timestamp("="*60)

    for i, (asin, item) in enumerate(samples[10:15]):
        log_with_timestamp(f"\n--- Sample {i+1}: ASIN={asin} ---")

        # Title
        raw_title = item.get('title', '')
        cleaned_title = clean_data(raw_title)
        if raw_title != cleaned_title:
            log_with_timestamp(f"  TITLE (changed):")
            log_with_timestamp(f"    RAW ({len(raw_title)}): {repr(raw_title[:200])}")
            log_with_timestamp(f"    CLEANED ({len(cleaned_title)}): {repr(cleaned_title[:200])}")

        # Description
        raw_desc = item.get('description', '')
        if isinstance(raw_desc, list):
            raw_desc = ' '.join(str(x) for x in raw_desc if x)
        cleaned_desc = clean_data(raw_desc)
        if raw_desc != cleaned_desc:
            log_with_timestamp(f"  DESCRIPTION (changed):")
            log_with_timestamp(f"    RAW ({len(raw_desc)}): {repr(raw_desc[:300])}")
            log_with_timestamp(f"    CLEANED ({len(cleaned_desc)}): {repr(cleaned_desc[:300])}")

        # Features
        raw_feat = item.get('features', [])
        if isinstance(raw_feat, list):
            raw_feat_str = ' '.join(str(x) for x in raw_feat if x)
        else:
            raw_feat_str = str(raw_feat)
        cleaned_feat = clean_data(raw_feat)
        if raw_feat_str != cleaned_feat:
            log_with_timestamp(f"  FEATURES (changed):")
            log_with_timestamp(f"    RAW ({len(raw_feat_str)}): {repr(raw_feat_str[:200])}")
            log_with_timestamp(f"    CLEANED ({len(cleaned_feat)}): {repr(cleaned_feat[:200])}")


def analyze_built_document(samples: list) -> None:
    """Analyze the fully built document text"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("BUILT DOCUMENT ANALYSIS")
    log_with_timestamp("="*60)

    for i, (asin, item) in enumerate(samples[15:20]):
        doc = build_stark_document(item, None, add_rel=False, compact=False)

        log_with_timestamp(f"\n--- Built Document {i+1}: ASIN={asin} ---")
        log_with_timestamp(f"  Total chars: {len(doc)}")
        log_with_timestamp(f"  Total tokens (approx): {len(doc.split())}")
        log_with_timestamp(f"  Full document:\n{doc}")
        log_with_timestamp("  " + "-"*40)


def find_problematic_documents(samples: list) -> None:
    """Find documents with specific problems"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("PROBLEMATIC DOCUMENT SEARCH")
    log_with_timestamp("="*60)

    problematic = []

    for asin, item in samples:
        issues = []

        # Check description
        desc = item.get('description', '')
        if isinstance(desc, list):
            desc = ' '.join(str(x) for x in desc if x)
        elif not isinstance(desc, str):
            desc = str(desc)

        if not desc or len(desc.strip()) < 10:
            issues.append("empty_or_short_description")
        elif len(desc) > 8000:
            issues.append(f"very_long_description_{len(desc)}")
        if '<' in desc and '>' in desc and not issues:
            issues.append("has_html_tags")
        if 'function' in desc.lower() or '<script' in desc.lower():
            issues.append("has_script_code")
        if re.search(r'\\x[0-9a-fA-F]{2}', desc):
            issues.append("has_unicode_escapes")

        # Check features
        features = item.get('features', [])
        if not features or (isinstance(features, list) and len(features) == 0):
            issues.append("no_features")

        if issues:
            problematic.append((asin, issues, desc[:200] if desc else ""))

    log_with_timestamp(f"Found {len(problematic)} potentially problematic documents")

    if problematic:
        log_with_timestamp("\nTop 5 problematic documents:")
        for asin, issues, preview in problematic[:5]:
            log_with_timestamp(f"\n  ASIN: {asin}")
            log_with_timestamp(f"  Issues: {', '.join(issues)}")
            log_with_timestamp(f"  Description preview: {repr(preview[:150])}")


def analyze_features_structure(samples: list) -> None:
    """Analyze the structure of features field in 2023 data"""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("FEATURES FIELD STRUCTURE ANALYSIS (2023)")
    log_with_timestamp("="*60)

    feature_types = Counter()
    sample_features = []

    for asin, item in samples:
        features = item.get('features', [])
        if features:
            sample_features.append((asin, features[:3]))  # First 3 features
            if isinstance(features[0], dict):
                # 2023 format: [{"value": [...]}, ...]
                feature_types['dict_with_value'] += 1
                # Check structure
                f0 = features[0]
                if 'value' in f0:
                    feature_types['has_value_key'] += 1
            elif isinstance(features[0], str):
                feature_types['string_list'] += 1
            elif isinstance(features[0], list):
                feature_types['nested_list'] += 1
            else:
                feature_types['other'] += 1

    log_with_timestamp(f"\nFeature structure types:")
    for ftype, count in feature_types.most_common():
        log_with_timestamp(f"  {ftype}: {count} ({count/len(samples)*100:.1f}%)")

    log_with_timestamp(f"\nSample features (first 3 items):")
    for asin, feats in sample_features[:5]:
        log_with_timestamp(f"\n  ASIN {asin}:")
        for j, f in enumerate(feats):
            log_with_timestamp(f"    [{j}]: {repr(f)[:150]}")


def main():
    meta_file = "/fs04/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz"

    log_with_timestamp("="*60)
    log_with_timestamp("DOCUMENT CONTENT DEBUG SCRIPT")
    log_with_timestamp("="*60)
    log_with_timestamp(f"Meta file: {meta_file}")

    # Load raw samples (more samples for better analysis)
    samples = load_raw_metadata(meta_file, n_samples=5000)

    # Run all analyses
    analyze_raw_content(samples)
    show_detailed_raw_samples(samples)
    compare_before_after_cleaning(samples)
    analyze_built_document(samples)
    find_problematic_documents(samples)
    analyze_features_structure(samples)

    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("DEBUG COMPLETE")
    log_with_timestamp("="*60)


if __name__ == "__main__":
    main()
