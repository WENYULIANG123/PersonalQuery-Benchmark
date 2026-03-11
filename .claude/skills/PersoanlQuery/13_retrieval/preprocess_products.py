#!/usr/bin/env python3
"""
Preprocess product data with STaRK-style cleaning.
Saves cleaned data to file for fast loading in subsequent evaluations.
"""

import json
import os
import gzip
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from utils import (
    log_with_timestamp,
    clean_data,
    process_brand,
    decode_html_entities,
    load_product_metadata,
    load_reviews_for_products,
    load_qa_for_products,
)

BASE_DIR = "/home/wlia0047/ar57/wenyu"
CACHE_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval/cache")
CATEGORY = "Arts_Crafts_and_Sewing"

META_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"meta_{CATEGORY}.json")
REVIEW_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"{CATEGORY}.json.gz")
QA_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"qa_{CATEGORY}.json.gz")

MAX_REVIEWS_PER_PRODUCT = 25
MAX_QA_PER_PRODUCT = 25


def preprocess_all_products():
    """Load, clean, and save all product data."""
    log_with_timestamp("=" * 60)
    log_with_timestamp("Preprocessing Product Data (STaRK-style cleaning)")
    log_with_timestamp("=" * 60)

    os.makedirs(CACHE_DIR, exist_ok=True)

    log_with_timestamp(f"Meta file: {META_FILE}")
    log_with_timestamp(f"Review file: {REVIEW_FILE}")
    log_with_timestamp(f"Q&A file: {QA_FILE}")
    log_with_timestamp(f"Cache dir: {CACHE_DIR}")

    all_asins = None
    products = {}
    all_metadata = {}

    if os.path.exists(META_FILE):
        log_with_timestamp("[1/4] Loading product metadata...")
        products, all_metadata = load_product_metadata(META_FILE, None)
        all_asins = set(products.keys())
        log_with_timestamp(f"  Loaded {len(products)} products, {len(all_metadata)} total")

    if os.path.exists(REVIEW_FILE) and all_asins:
        log_with_timestamp("[2/4] Loading and cleaning reviews...")
        products = load_reviews_for_products(
            REVIEW_FILE, products,
            max_reviews_per_product=MAX_REVIEWS_PER_PRODUCT,
            min_review_words=0
        )

    if os.path.exists(QA_FILE) and all_asins:
        log_with_timestamp("[3/4] Loading and cleaning Q&A...")
        products = load_qa_for_products(
            QA_FILE, products,
            max_qa_per_product=MAX_QA_PER_PRODUCT
        )

    log_with_timestamp("[4/4] Saving cleaned data...")

    cache_file = os.path.join(CACHE_DIR, f"cleaned_products_{CATEGORY}.pkl")
    cache_meta_file = os.path.join(CACHE_DIR, f"cleaned_metadata_{CATEGORY}.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump(products, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_with_timestamp(f"  Saved products: {cache_file}")

    with open(cache_meta_file, 'wb') as f:
        pickle.dump(all_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_with_timestamp(f"  Saved metadata: {cache_meta_file}")

    stats = {
        'total_products': len(products),
        'total_metadata': len(all_metadata),
        'products_with_reviews': sum(1 for p in products.values() if p.get('reviews')),
        'total_reviews': sum(len(p.get('reviews', [])) for p in products.values()),
        'products_with_qa': sum(1 for p in products.values() if p.get('qa')),
        'total_qa': sum(len(p.get('qa', [])) for p in products.values()),
    }

    stats_file = os.path.join(CACHE_DIR, f"cleaned_stats_{CATEGORY}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    log_with_timestamp(f"  Saved stats: {stats_file}")

    log_with_timestamp("")
    log_with_timestamp("=" * 60)
    log_with_timestamp("PREPROCESSING COMPLETE")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"Total products: {stats['total_products']}")
    log_with_timestamp(f"Products with reviews: {stats['products_with_reviews']}")
    log_with_timestamp(f"Total reviews: {stats['total_reviews']}")
    log_with_timestamp(f"Products with Q&A: {stats['products_with_qa']}")
    log_with_timestamp(f"Total Q&A: {stats['total_qa']}")
    log_with_timestamp("")
    log_with_timestamp("Files created:")
    log_with_timestamp(f"  {cache_file}")
    log_with_timestamp(f"  {cache_meta_file}")
    log_with_timestamp(f"  {stats_file}")

    return products, all_metadata


if __name__ == "__main__":
    preprocess_all_products()
