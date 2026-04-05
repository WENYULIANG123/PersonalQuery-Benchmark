#!/usr/bin/env python3
"""
Stage 5: 收集用户长句数据
直接复用 Stage 0 的输出（selected_users.json + reviews_*.json）
收集每个用户10句20词以上的句子，并计算14维语言复杂度特征
并进行回译（英->中->英）生成语义相同但表达不同的句子
"""
import json
import os
import re
import glob
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import spacy
from deep_translator import GoogleTranslator


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def has_long_sentence(text: str, min_words: int) -> bool:
    """Check if text contains at least one sentence with >= min_words words."""
    if not text:
        return False
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if len(sent.split()) >= min_words:
            return True
    return False


def extract_long_sentences(text: str, min_words: int) -> List[str]:
    """Extract all sentences with >= min_words words."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sent.strip() for sent in sentences if len(sent.split()) >= min_words]


def count_long_sentences(text: str, min_words: int) -> int:
    """Count how many sentences have >= min_words words."""
    if not text:
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sum(1 for sent in sentences if len(sent.split()) >= min_words)


def extract_14d_features(text: str, nlp) -> Dict[str, float]:
    """从文本提取14维复杂度特征（spaCy计算）"""
    doc = nlp(text)
    sentences = list(doc.sents)

    if not sentences:
        return {
            'subordinate_clause_freq': 0.0,
            'dep_distance': 0.0,
            'modifier_density': 0.0,
            'coord_chain': 0.0,
            'negation_scope': 0.0,
            'voice_ratio': 0.0,
            'branching_direction': 0.5,
            'advcl_freq': 0.0,
            'comp_clause_freq': 0.0,
            'fanout': 0.0,
            'parataxis_freq': 0.0,
            'prep_density': 0.0,
            'appos_freq': 0.0,
            'word_count': 0,
            'avg_sentence_length': 0.0,
        }

    all_features = []
    for sent in sentences:
        tokens = [t for t in sent if not t.is_space and not t.is_punct]
        n = len(tokens)
        if n == 0:
            continue

        # 1. 从句深度
        SUBORDINATE_DEPS = {'acl', 'acl:relcl', 'relcl', 'csubj', 'csubjpass'}
        clause_depth = sum(1 for t in tokens if t.dep_ in SUBORDINATE_DEPS) / n

        # 2. 依存距离
        total_dist = sum(abs(t.i - t.head.i) for t in tokens if t.head != t and t.head.i < len(tokens))
        max_possible = n * (n - 1) / 2 if n > 1 else 1
        dep_distance = total_dist / max_possible if max_possible > 0 else 0.0

        # 3. 修饰语密度
        MODIFIER_DEPS = {'amod', 'advmod', 'compound', 'nn', 'nummod', 'poss'}
        modifier_density = sum(1 for t in tokens if t.dep_ in MODIFIER_DEPS) / n

        # 4. 并列链
        coord_chain = min(sum(1 for t in tokens if t.dep_ == 'conj'), 5) / 5.0

        # 5. 否定作用域
        neg_scope = 0
        for t in tokens:
            if t.dep_ == 'neg':
                subtree = len(list(t.subtree))
                neg_scope = max(neg_scope, subtree)
        negation_scope = min(neg_scope / n, 1.0) if n > 0 else 0.0

        # 6. Voice Ratio
        nsubj_count = sum(1 for t in tokens if t.dep_ == 'nsubj')
        nsubjpass_count = sum(1 for t in tokens if t.dep_ == 'nsubjpass')
        total_voice = nsubj_count + nsubjpass_count
        voice_ratio = nsubjpass_count / total_voice if total_voice > 0 else 0.0

        # 7. Branching Direction
        left_children = sum(1 for t in tokens if len(list(t.lefts)) > 0)
        right_children = sum(1 for t in tokens if len(list(t.rights)) > 0)
        total_branches = left_children + right_children
        branching_direction = left_children / total_branches if total_branches > 0 else 0.5

        # 8. Relative Clause Frequency
        RELCL_DEPS = {'relcl', 'acl:relcl'}
        relcl_count = sum(1 for t in tokens if t.dep_ in RELCL_DEPS)
        relative_clause_freq = relcl_count / n if n > 0 else 0.0

        # 9. Adverbial Clause Frequency
        advcl_count = sum(1 for t in tokens if t.dep_ == 'advcl')
        advcl_freq = advcl_count / n if n > 0 else 0.0

        # 10. Complement Clause Frequency
        COMP_DEPS = {'ccomp', 'xcomp'}
        comp_count = sum(1 for t in tokens if t.dep_ in COMP_DEPS)
        comp_clause_freq = comp_count / n if n > 0 else 0.0

        # 11. Dependency Tree Fan-out
        children_count = Counter(t.head.i for t in tokens if t.dep_ != 'ROOT')
        fanout = sum(children_count.values()) / len(children_count) if children_count else 0.0

        # 12. Parataxis Frequency
        para_count = sum(1 for t in tokens if t.dep_ == 'parataxis')
        parataxis_freq = para_count / n if n > 0 else 0.0

        # 13. Prepositional Phrase Density
        prep_count = sum(1 for t in tokens if t.dep_ == 'prep')
        prep_density = prep_count / n if n > 0 else 0.0

        # 14. Apposition Frequency
        appos_count = sum(1 for t in tokens if t.dep_ == 'appos')
        appos_freq = appos_count / n if n > 0 else 0.0

        all_features.append({
            'clause_depth': clause_depth,
            'relative_clause_freq': relative_clause_freq,
            'dep_distance': dep_distance,
            'modifier_density': modifier_density,
            'coord_chain': coord_chain,
            'negation_scope': negation_scope,
            'voice_ratio': voice_ratio,
            'branching_direction': branching_direction,
            'advcl_freq': advcl_freq,
            'comp_clause_freq': comp_clause_freq,
            'fanout': fanout,
            'parataxis_freq': parataxis_freq,
            'prep_density': prep_density,
            'appos_freq': appos_freq,
            'word_count': n,
        })

    if not all_features:
        return {
            'subordinate_clause_freq': 0.0, 'dep_distance': 0.0,
            'modifier_density': 0.0, 'coord_chain': 0.0,
            'negation_scope': 0.0, 'voice_ratio': 0.0,
            'branching_direction': 0.5, 'advcl_freq': 0.0,
            'comp_clause_freq': 0.0, 'fanout': 0.0,
            'parataxis_freq': 0.0, 'prep_density': 0.0,
            'appos_freq': 0.0, 'word_count': 0, 'avg_sentence_length': 0.0,
        }

    n_sents = len(all_features)
    total_words = sum(f.get('word_count', 0) for f in all_features)
    return {
        'subordinate_clause_freq': round(sum((f.get('clause_depth', 0) + f.get('relative_clause_freq', 0)) / 2 for f in all_features) / n_sents, 6),
        'dep_distance': round(sum(f.get('dep_distance', 0) for f in all_features) / n_sents, 6),
        'modifier_density': round(sum(f.get('modifier_density', 0) for f in all_features) / n_sents, 6),
        'coord_chain': round(sum(f.get('coord_chain', 0) for f in all_features) / n_sents, 6),
        'negation_scope': round(sum(f.get('negation_scope', 0) for f in all_features) / n_sents, 6),
        'voice_ratio': round(sum(f.get('voice_ratio', 0) for f in all_features) / n_sents, 6),
        'branching_direction': round(sum(f.get('branching_direction', 0.5) for f in all_features) / n_sents, 6),
        'advcl_freq': round(sum(f.get('advcl_freq', 0) for f in all_features) / n_sents, 6),
        'comp_clause_freq': round(sum(f.get('comp_clause_freq', 0) for f in all_features) / n_sents, 6),
        'fanout': round(sum(f.get('fanout', 0) for f in all_features) / n_sents, 6),
        'parataxis_freq': round(sum(f.get('parataxis_freq', 0) for f in all_features) / n_sents, 6),
        'prep_density': round(sum(f.get('prep_density', 0) for f in all_features) / n_sents, 6),
        'appos_freq': round(sum(f.get('appos_freq', 0) for f in all_features) / n_sents, 6),
        'word_count': total_words,
        'avg_sentence_length': round(total_words / n_sents, 1),
    }


def backtranslate_en2zh2en(sentence: str, max_retries: int = 3) -> Optional[str]:
    """回译：英文 -> 中文 -> 英文 (使用 Google Translate)"""
    for attempt in range(max_retries):
        try:
            # Step 1: 英文 -> 中文 (简体中文)
            chinese_text = GoogleTranslator(source='en', target='zh-CN').translate(sentence)
            if not chinese_text:
                continue

            time.sleep(0.5)  # 避免请求过快

            # Step 2: 中文 -> 英文
            english_back = GoogleTranslator(source='zh-CN', target='en').translate(chinese_text)
            if not english_back:
                continue

            return english_back
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            log_with_timestamp(f"  [Warning] Back-translation failed: {e}")
            return None

    return None


def main() -> None:
    # ============ 硬编码参数 ============
    STAGE0_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
    OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_sentences"
    TARGET_SENTENCES = 10     # 每个用户收集的句子数
    MAX_USERS = 10            # 最大用户数

    import shutil
    if os.path.isdir(OUTPUT_DIR):
        log_with_timestamp(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 5: Collect User Long Sentences (from Stage 0)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Stage 0 directory: {STAGE0_DIR}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Target sentences per user: {TARGET_SENTENCES}")
    log_with_timestamp(f"Max users: {MAX_USERS}")

    # 直接从 Stage 0 的 selected_users.json 读取用户列表
    selected_users_file = os.path.join(STAGE0_DIR, "selected_users.json")
    log_with_timestamp(f"Loading selected users from: {selected_users_file}")
    with open(selected_users_file, "r", encoding="utf-8") as f:
        stage0_data = json.load(f)

    selected_users = stage0_data.get("selected_users", [])
    if not selected_users:
        log_with_timestamp("No users found in Stage 0 output. Exiting.")
        return

    # 取前 MAX_USERS 个用户（按 long_sentence_count 降序）
    selected_users = sorted(selected_users, key=lambda x: x.get("long_sentence_count", 0), reverse=True)[:MAX_USERS]
    selected_user_ids = [u["user_id"] for u in selected_users]
    user_counts = {u["user_id"]: u["long_sentence_count"] for u in selected_users}
    user_product_counts = {u["user_id"]: u.get("product_count", 0) for u in selected_users}

    log_with_timestamp(f"Selected {len(selected_user_ids)} users from Stage 0")

    # 加载spaCy模型用于14维特征提取
    log_with_timestamp("Loading spaCy model for 14D feature extraction...")
    nlp = spacy.load("en_core_web_sm")

    # 收集所有用户数据
    all_users_data = []
    success_count = 0

    for idx, user_id in enumerate(selected_user_ids, start=1):
        log_with_timestamp(f"[{idx}/{len(selected_user_ids)}] Processing user {user_id}...")

        # 直接读取 Stage 0 输出的用户评论文件
        user_review_file = os.path.join(STAGE0_DIR, f"reviews_{user_id}.json")
        if not os.path.exists(user_review_file):
            log_with_timestamp(f"  User file not found: {user_review_file}, skip")
            continue

        with open(user_review_file, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        # 从 target_reviews 中提取长句子
        all_long_sentences = []
        results = user_data.get("results", [])
        for result in results:
            target_reviews = result.get("target_reviews", [])
            for review_text in target_reviews:
                long_sents = extract_long_sentences(review_text, 20)  # MIN_WORDS = 20
                all_long_sentences.extend(long_sents)

        if len(all_long_sentences) < TARGET_SENTENCES:
            log_with_timestamp(f"  User {user_id} only has {len(all_long_sentences)} long sentences, skip")
            continue

        selected_sents = all_long_sentences[:TARGET_SENTENCES]

        # 为每个句子计算14维特征，并进行回译
        sentences_with_features = []
        for sent_idx, sent in enumerate(selected_sents):
            log_with_timestamp(f"  [{sent_idx+1}/{len(selected_sents)}] Processing sentence...")
            features = extract_14d_features(sent, nlp)

            # 回译：英->中->英
            log_with_timestamp(f"    Back-translating...")
            backtrans_sent = backtranslate_en2zh2en(sent)

            # 为回译句子计算14维特征
            backtrans_features = None
            if backtrans_sent:
                backtrans_features = extract_14d_features(backtrans_sent, nlp)
                log_with_timestamp(f"    Original: {sent[:60]}...")
                log_with_timestamp(f"    Backtrans: {backtrans_sent[:60]}...")

            sentence_data = {
                "sentence": sent,
                "features_14d": features,
            }
            if backtrans_sent:
                sentence_data["backtrans_sentence"] = backtrans_sent
                sentence_data["backtrans_features_14d"] = backtrans_features

            sentences_with_features.append(sentence_data)

        all_users_data.append({
            "user_id": user_id,
            "long_sentence_count": user_counts[user_id],
            "product_count": user_product_counts[user_id],
            "sentences": sentences_with_features,
        })
        success_count += 1

    # 保存到合并文件
    merged_output = {
        "timestamp": datetime.now().isoformat(),
        "source": "Stage 0 output",
        "stage0_dir": STAGE0_DIR,
        "selection_criteria": {
            "min_words": 20,
            "min_long_sentences": 10,
            "target_sentences_per_user": TARGET_SENTENCES,
            "max_users": MAX_USERS,
        },
        "total_selected": len(selected_user_ids),
        "successfully_collected": success_count,
        "users": all_users_data,
    }

    output_file = os.path.join(OUTPUT_DIR, "all_users_merged.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_output, f, indent=2, ensure_ascii=False)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 5 Complete")
    log_with_timestamp(f"Selected users: {len(selected_user_ids)}")
    log_with_timestamp(f"Successfully collected: {success_count}")
    log_with_timestamp(f"Output file: {output_file}")
    log_with_timestamp("=" * 80)
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
