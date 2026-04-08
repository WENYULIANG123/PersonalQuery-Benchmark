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

# 翻译模型
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    """本地翻译模型"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_with_timestamp(f"Translator device: {self.device}")

        log_with_timestamp("Loading EN->ZH model...")
        self.en_zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        self.en_zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(self.device)

        log_with_timestamp("Loading ZH->EN model...")
        self.zh_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        self.zh_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """翻译单个文本"""
        if src_lang == "en" and tgt_lang == "zh":
            tokenizer, model = self.en_zh_tokenizer, self.en_zh_model
        elif src_lang == "zh" and tgt_lang == "en":
            tokenizer, model = self.zh_en_tokenizer, self.zh_en_model
        else:
            raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            log_with_timestamp(f"  [Warning] Translation error: {e}")
            return None

    def translate_batch(self, texts: list, src_lang: str, tgt_lang: str) -> list:
        """批量翻译文本"""
        if src_lang == "en" and tgt_lang == "zh":
            tokenizer, model = self.en_zh_tokenizer, self.en_zh_model
        elif src_lang == "zh" and tgt_lang == "en":
            tokenizer, model = self.zh_en_tokenizer, self.zh_en_model
        else:
            raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)

            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return results
        except Exception as e:
            log_with_timestamp(f"  [Warning] Batch translation error: {e}")
            return [None] * len(texts)

    def backtranslate_batch(self, texts: list) -> list:
        """批量回译：EN -> ZH -> EN"""
        # Step 1: EN -> ZH
        zh_texts = self.translate_batch(texts, "en", "zh")
        if not zh_texts or all(t is None for t in zh_texts):
            return [None] * len(texts)

        # Step 2: ZH -> EN
        en_texts = self.translate_batch(zh_texts, "zh", "en")
        return en_texts


# 全局翻译器实例
_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


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


def backtranslate_en2zh2en(sentence: str) -> Optional[str]:
    """回译单个句子：英文 -> 中文 -> 英文 (使用本地翻译模型)"""
    translator = get_translator()
    return translator.backtranslate(sentence)


def backtranslate_batch(sentences: list) -> list:
    """批量回译：英文 -> 中文 -> 英文 (使用本地翻译模型)"""
    translator = get_translator()
    return translator.backtranslate_batch(sentences)


def main() -> None:
    # ============ 硬编码参数 ============
    STAGE0_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
    OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_sentences"
    TARGET_SENTENCES = 10     # 每个用户收集的句子数
    MAX_USERS = None          # 最大用户数（None表示处理所有用户）

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

    # 加载翻译模型
    log_with_timestamp("Loading translation models...")
    translator = get_translator()

    # 串行处理所有用户
    all_users_data = []
    success_count = 0

    # 每批处理10个用户（100个句子）
    BATCH_USERS = 1
    BATCH_SENTENCES = BATCH_USERS * 10  # 100

    for batch_start in range(0, len(selected_user_ids), BATCH_USERS):
        batch_end = min(batch_start + BATCH_USERS, len(selected_user_ids))
        batch_user_ids = selected_user_ids[batch_start:batch_end]
        batch_idx = batch_start + 1
        log_with_timestamp(f"[{batch_idx}/{len(selected_user_ids)}] Processing users {batch_start+1}-{batch_end} ({len(batch_user_ids)} users)...")

        # 收集本批次所有用户的句子
        batch_sents = []  # 所有句子
        batch_user_indices = []  # 每个句子属于哪个用户(0-9)
        batch_user_sents_count = []  # 每个用户收集到的句子数

        for user_id in batch_user_ids:
            user_review_file = os.path.join(STAGE0_DIR, f"reviews_{user_id}.json")
            if not os.path.exists(user_review_file):
                batch_user_sents_count.append(0)
                continue

            with open(user_review_file, "r", encoding="utf-8") as f:
                user_data = json.load(f)

            all_long_sentences = []
            results = user_data.get("results", [])
            for result in results:
                target_reviews = result.get("target_reviews", [])
                for review_text in target_reviews:
                    long_sents = extract_long_sentences(review_text, 20)
                    all_long_sentences.extend(long_sents)

            selected_sents = all_long_sentences[:10] if len(all_long_sentences) >= 10 else all_long_sentences
            batch_user_sents_count.append(len(selected_sents))

            for sent in selected_sents:
                batch_sents.append(sent)
                batch_user_indices.append(len(batch_user_sents_count) - 1)

        # 跳过句子数不足的用户
        valid_mask = [count >= 10 for count in batch_user_sents_count]
        if not any(valid_mask):
            log_with_timestamp(f"  No valid users in this batch, skip")
            continue

        # 批量计算原句特征
        orig_features_list = [extract_14d_features(sent, nlp) for sent in batch_sents]

        # 批量回译所有句子
        backtrans_sents = backtranslate_batch(batch_sents)

        # 批量计算回译句特征
        backtrans_features_list = []
        for bt in backtrans_sents:
            if bt:
                backtrans_features_list.append(extract_14d_features(bt, nlp))
            else:
                backtrans_features_list.append(None)

        # 按用户组装结果
        user_idx = 0
        sent_idx = 0
        for user_id in batch_user_ids:
            if batch_user_sents_count[user_idx] < 10:
                user_idx += 1
                continue

            num_sents = batch_user_sents_count[user_idx]
            user_sents = batch_sents[sent_idx:sent_idx + num_sents]
            user_orig_features = orig_features_list[sent_idx:sent_idx + num_sents]
            user_bt_sents = backtrans_sents[sent_idx:sent_idx + num_sents]
            user_bt_features = backtrans_features_list[sent_idx:sent_idx + num_sents]

            sentences_with_features = []
            for i, sent in enumerate(user_sents):
                sentence_data = {
                    "sentence": sent,
                    "features_14d": user_orig_features[i],
                }
                if user_bt_sents[i]:
                    sentence_data["backtrans_sentence"] = user_bt_sents[i]
                    sentence_data["backtrans_features_14d"] = user_bt_features[i]
                sentences_with_features.append(sentence_data)

            all_users_data.append({
                "user_id": user_id,
                "long_sentence_count": user_counts[user_id],
                "product_count": user_product_counts[user_id],
                "sentences": sentences_with_features,
            })
            success_count += 1

            sent_idx += num_sents
            user_idx += 1

        log_with_timestamp(f"  -> Completed users {batch_start+1}-{batch_end}")

    # 保存到合并文件
    merged_output = {
        "timestamp": datetime.now().isoformat(),
        "source": "Stage 0 output",
        "stage0_dir": STAGE0_DIR,
        "selection_criteria": {
            "min_words": 20,
            "min_long_sentences": 10,
            "target_sentences_per_user": 10,
            "max_users": MAX_USERS,
            "batch_size_users": BATCH_USERS,
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
