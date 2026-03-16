#!/usr/bin/env python3
"""
Optimized preprocessing using pandas and pickle (STaRK-style).
"""

import json
import os
import gzip
import pickle
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = "/home/wlia0047/ar57/wenyu"
CACHE_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval/cache")
CATEGORY = "Arts_Crafts_and_Sewing"

META_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"meta_{CATEGORY}.json")
REVIEW_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"{CATEGORY}.json.gz")
QA_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw", f"qa_{CATEGORY}.json.gz")

MAX_REVIEWS_PER_PRODUCT = 25
MAX_QA_PER_PRODUCT = 25


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def clean_brand(brand):
    if not brand:
        return ''
    import string
    brand = str(brand).strip(" \" .*+,-_!@#$%^&*();\/|<>'\t\n\r\\")
    if brand.startswith('by '):
        brand = brand[3:]
    if brand.endswith('.com'):
        brand = brand[:-4]
    if brand.startswith('www.'):
        brand = brand[4:]
    if len(brand) > 100:
        brand = brand.split(' ')[0]
    return brand.strip()


def clean_text(text):
    if text is None:
        return ''
    text = str(text)
    if '<' in text and '>' in text:
        from bs4 import BeautifulSoup
        text = ' '.join(BeautifulSoup(text, "lxml").text.split())
    else:
        text = ' '.join(text.split())
    return text.strip()


def clean_list_field(items):
    """Clean a list of text items"""
    if not items:
        return []
    if isinstance(items, list):
        return [clean_text(x) for x in items if x]
    return [clean_text(items)]


def load_meta_as_df(meta_file):
    log(f"Loading meta: {meta_file}")
    records = []
    open_func = gzip.open if meta_file.endswith('.gz') else open
    with open_func(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                records.append({
                    'asin': item.get('asin'),
                    'title': clean_text(item.get('title', '')),
                    'brand': clean_brand(item.get('brand', '')),
                    'category': item.get('category', []),
                    'feature': clean_list_field(item.get('feature', [])),
                    'description': clean_list_field(item.get('description', [])),
                    'rank': clean_text(item.get('rank', '')),
                    'also_buy': item.get('also_buy', []),
                    'also_view': item.get('also_view', []),
                })
            except:
                continue
    df = pd.DataFrame(records)
    log(f"  Loaded {len(df)} products")
    return df


def load_reviews_as_df(review_file):
    log(f"Loading reviews: {review_file}")
    open_func = gzip.open if review_file.endswith('.gz') else open
    records = []
    with open_func(review_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                records.append({
                    'asin': item.get('asin'),
                    'summary': item.get('summary', ''),
                    'reviewText': item.get('reviewText', ''),
                    'vote': item.get('vote'),
                    'overall': item.get('overall'),
                })
            except:
                continue
    df = pd.DataFrame(records)
    log(f"  Loaded {len(df)} reviews")
    return df


def load_qa_as_df(qa_file):
    log(f"Loading Q&A: {qa_file}")
    open_func = gzip.open if qa_file.endswith('.gz') else open
    
    def parse_line(line):
        line = line.strip()
        if not line:
            return None
        try:
            return eval(line)
        except:
            return None
    
    records = []
    with open_func(qa_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                item = parse_line(line)
                if item and item.get('question') and item.get('answer'):
                    records.append({
                        'asin': item.get('asin'),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                    })
            except:
                continue
    df = pd.DataFrame(records)
    log(f"  Loaded {len(df)} Q&A entries")
    return df


def preprocess():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    log("=" * 60)
    log("OPTIMIZED PREPROCESSING (pandas + pickle)")
    log("=" * 60)
    
    cache_meta = os.path.join(CACHE_DIR, f"df_meta_{CATEGORY}.pkl")
    cache_review = os.path.join(CACHE_DIR, f"df_review_{CATEGORY}.pkl")
    cache_qa = os.path.join(CACHE_DIR, f"df_qa_{CATEGORY}.pkl")
    
    # Load or create meta
    if os.path.exists(cache_meta):
        log("Loading cached meta...")
        df_meta = pd.read_pickle(cache_meta)
    else:
        df_meta = load_meta_as_df(META_FILE)
        df_meta.to_pickle(cache_meta)
    
    # Load or create reviews
    if os.path.exists(cache_review):
        log("Loading cached reviews...")
        df_review = pd.read_pickle(cache_review)
    else:
        df_review = load_reviews_as_df(REVIEW_FILE)
        df_review.to_pickle(cache_review)
    
    # Load or create Q&A
    if os.path.exists(cache_qa):
        log("Loading cached Q&A...")
        df_qa = pd.read_pickle(cache_qa)
    else:
        df_qa = load_qa_as_df(QA_FILE)
        df_qa.to_pickle(cache_qa)
    
    # Merge and filter
    log("Merging data...")
    
    # Get unique ASINs that have reviews
    asins_with_reviews = df_review['asin'].unique()
    df_meta_with_reviews = df_meta[df_meta['asin'].isin(asins_with_reviews)].copy()
    
    log(f"Products with reviews: {len(df_meta_with_reviews)}")
    
    # Process reviews - keep top by vote
    log("Processing reviews...")
    df_review['vote_num'] = df_review['vote'].fillna(0).astype(str).str.replace(',', '').astype(float)
    df_review = df_review.sort_values('vote_num', ascending=False)
    df_review = df_review.groupby('asin').head(MAX_REVIEWS_PER_PRODUCT)
    
    # Clean review text
    df_review['summary'] = df_review['summary'].fillna('').apply(clean_text)
    df_review['reviewText'] = df_review['reviewText'].fillna('').apply(clean_text)
    
    # Process Q&A
    log("Processing Q&A...")
    df_qa = df_qa.groupby('asin').head(MAX_QA_PER_PRODUCT)
    
    # Clean Q&A text
    df_qa['question'] = df_qa['question'].fillna('').apply(clean_text)
    df_qa['answer'] = df_qa['answer'].fillna('').apply(clean_text)
    
    # Save processed data
    log("Saving processed data...")
    df_meta_with_reviews.to_pickle(os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl"))
    df_review.to_pickle(os.path.join(CACHE_DIR, f"reviews_{CATEGORY}.pkl"))
    df_qa.to_pickle(os.path.join(CACHE_DIR, f"qa_{CATEGORY}.pkl"))
    
    # Stats
    stats = {
        'total_products': len(df_meta_with_reviews),
        'total_reviews': len(df_review),
        'total_qa': len(df_qa),
    }
    with open(os.path.join(CACHE_DIR, f"stats_{CATEGORY}.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    log("=" * 60)
    log("DONE!")
    log(f"Products: {stats['total_products']}")
    log(f"Reviews: {stats['total_reviews']}")
    log(f"Q&A: {stats['total_qa']}")
    log("=" * 60)


if __name__ == "__main__":
    preprocess()
