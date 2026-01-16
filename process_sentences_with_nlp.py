#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ontology-nlp 项目的 NLP 预处理代码处理评论中的每一句话
"""

import json
import sys
import re
import nltk
from nltk.tokenize import sent_tokenize

# 添加 ontology-nlp 项目路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/ontology-nlp/ontology-nlp-master')

# 尝试导入 ontology-nlp 的 util 模块
try:
    from util import (
        tokenization,
        normalization,
        removing_stop_words,
        get_stemming,
        BASE,
        TOKENIZATION,
        NORMALIZATION,
        REMOVE_STOP_WORDS,
        PORTER_STEMMER,
        SNOWBALL_STEMMER,
        LANCASTER_STEMMER,
        LEMMATIZER,
        LEMMATIZER_AND_POS
    )
    print("成功导入 ontology-nlp 的预处理函数")
except ImportError as e:
    print(f"导入 ontology-nlp 模块失败: {e}")
    print("将使用本地实现的预处理函数")
    # 如果导入失败，使用本地实现
    BASE = "B"
    TOKENIZATION = "T"
    NORMALIZATION = "N"
    REMOVE_STOP_WORDS = "R"
    PORTER_STEMMER = "SP"
    SNOWBALL_STEMMER = "SS"
    LANCASTER_STEMMER = "SL"
    LEMMATIZER = "L"
    LEMMATIZER_AND_POS = "LT"
    
    def change_to_snake_case(name):
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        name = pattern.sub(' ', name)
        return name
    
    def change_british_to_american(word):
        # 简化版本，不依赖 enchant
        return word
    
    def tokenization(name):
        name = re.sub(r'[^A-Za-z0-9]+', ' ', str(name))
        if " " not in name:
            name = change_to_snake_case(name)
        return name
    
    def normalization(name):
        name = name.lower()
        name = change_british_to_american(name)
        return name
    
    def removing_stop_words(entity):
        word_tokens = nltk.tokenize.word_tokenize(entity)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        keywords = [w for w in word_tokens if w not in stop_words]
        name = ' '.join(keywords)
        return name
    
    def penn2morphy(tag):
        if tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        if tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        if tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        if tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        return nltk.corpus.wordnet.NOUN
    
    def get_stemming(entity, method):
        keywords = entity.split()
        strs = []
        for name in keywords:
            if method == PORTER_STEMMER:
                stemmer = nltk.stem.PorterStemmer()
                name = stemmer.stem(name)
            if method == SNOWBALL_STEMMER:
                stemmer = nltk.stem.SnowballStemmer("english")
                name = stemmer.stem(name)
            if method == LANCASTER_STEMMER:
                stemmer = nltk.stem.LancasterStemmer()
                name = stemmer.stem(name)
            if method == LEMMATIZER:
                lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
                name = lmtzr.lemmatize(name)
            if method == LEMMATIZER_AND_POS:
                lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
                tokens = nltk.word_tokenize(name)
                tagged_tokens = nltk.pos_tag(tokens)
                if tagged_tokens:
                    word, tag = tagged_tokens[0]
                    name = lmtzr.lemmatize(name, pos=penn2morphy(tag))
            strs.append(name)
        return ' '.join(strs)

# 下载必要的 NLTK 数据
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载 NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("下载 NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("下载 NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("下载 NLTK POS tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)

def process_sentence(sentence):
    """
    对单个句子应用所有 NLP 预处理方法，并记录每个步骤的修改说明
    """
    if not sentence or not sentence.strip():
        return {}
    
    # 原始文本
    base = sentence.strip()
    changes = {}
    
    # B: Base - 原始文本（无修改）
    changes["B"] = "原始文本，未进行任何修改"
    
    # T: Tokenization - 分词
    tokenized = tokenization(base)
    if base != tokenized:
        changes["T"] = f"分词处理：将非字母数字字符替换为空格，处理驼峰命名（如存在）。原始文本长度: {len(base)} 字符，处理后: {len(tokenized)} 字符"
    else:
        changes["T"] = "分词处理：文本未发生变化"
    
    # N: Normalization - 标准化
    normalized = normalization(tokenized)
    if tokenized != normalized:
        # 检查是否有大小写变化
        lower_changed = tokenized.lower() != tokenized
        changes["N"] = "标准化处理：将所有字母转换为小写"
        if lower_changed:
            changes["N"] += "，并尝试将英式拼写转换为美式拼写（如适用）"
    else:
        changes["N"] = "标准化处理：文本未发生变化"
    
    # R: Remove Stop Words - 移除停用词
    stopwords_removed = removing_stop_words(normalized)
    normalized_words = nltk.tokenize.word_tokenize(normalized)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    removed_words = [w for w in normalized_words if w in stop_words]
    if removed_words:
        changes["R"] = f"移除停用词：移除了 {len(removed_words)} 个停用词（如：'the', 'a', 'is', 'in' 等）。移除的停用词包括: {', '.join(removed_words[:10])}" + (f" 等共 {len(removed_words)} 个" if len(removed_words) > 10 else "")
    else:
        changes["R"] = "移除停用词：未移除任何停用词（文本中不包含停用词）"
    
    # SP: Porter Stemmer - Porter 词干提取
    porter_stemmed = get_stemming(stopwords_removed, PORTER_STEMMER)
    if stopwords_removed != porter_stemmed:
        # 找出被修改的词
        original_words = stopwords_removed.split()
        stemmed_words = porter_stemmed.split()
        changed_words = []
        for orig, stem in zip(original_words, stemmed_words):
            if orig != stem:
                changed_words.append(f"{orig}→{stem}")
        if changed_words:
            changes["SP"] = f"Porter 词干提取：对 {len(changed_words)} 个词进行了词干提取。示例变化: {', '.join(changed_words[:5])}" + (" 等" if len(changed_words) > 5 else "")
        else:
            changes["SP"] = "Porter 词干提取：文本未发生变化"
    else:
        changes["SP"] = "Porter 词干提取：文本未发生变化"
    
    # SS: Snowball Stemmer - Snowball 词干提取
    snowball_stemmed = get_stemming(stopwords_removed, SNOWBALL_STEMMER)
    if stopwords_removed != snowball_stemmed:
        original_words = stopwords_removed.split()
        stemmed_words = snowball_stemmed.split()
        changed_words = []
        for orig, stem in zip(original_words, stemmed_words):
            if orig != stem:
                changed_words.append(f"{orig}→{stem}")
        if changed_words:
            changes["SS"] = f"Snowball 词干提取：对 {len(changed_words)} 个词进行了词干提取。示例变化: {', '.join(changed_words[:5])}" + (" 等" if len(changed_words) > 5 else "")
        else:
            changes["SS"] = "Snowball 词干提取：文本未发生变化"
    else:
        changes["SS"] = "Snowball 词干提取：文本未发生变化"
    
    # SL: Lancaster Stemmer - Lancaster 词干提取
    lancaster_stemmed = get_stemming(stopwords_removed, LANCASTER_STEMMER)
    if stopwords_removed != lancaster_stemmed:
        original_words = stopwords_removed.split()
        stemmed_words = lancaster_stemmed.split()
        changed_words = []
        for orig, stem in zip(original_words, stemmed_words):
            if orig != stem:
                changed_words.append(f"{orig}→{stem}")
        if changed_words:
            changes["SL"] = f"Lancaster 词干提取：对 {len(changed_words)} 个词进行了词干提取。示例变化: {', '.join(changed_words[:5])}" + (" 等" if len(changed_words) > 5 else "")
        else:
            changes["SL"] = "Lancaster 词干提取：文本未发生变化"
    else:
        changes["SL"] = "Lancaster 词干提取：文本未发生变化"
    
    # L: Lemmatizer - 词形还原
    lemmatized = get_stemming(stopwords_removed, LEMMATIZER)
    if stopwords_removed != lemmatized:
        original_words = stopwords_removed.split()
        lemmatized_words = lemmatized.split()
        changed_words = []
        for orig, lem in zip(original_words, lemmatized_words):
            if orig != lem:
                changed_words.append(f"{orig}→{lem}")
        if changed_words:
            changes["L"] = f"词形还原：对 {len(changed_words)} 个词进行了词形还原。示例变化: {', '.join(changed_words[:5])}" + (" 等" if len(changed_words) > 5 else "")
        else:
            changes["L"] = "词形还原：文本未发生变化"
    else:
        changes["L"] = "词形还原：文本未发生变化"
    
    # LT: Lemmatizer with POS - 带词性标注的词形还原
    lemmatized_pos = get_stemming(stopwords_removed, LEMMATIZER_AND_POS)
    if stopwords_removed != lemmatized_pos:
        original_words = stopwords_removed.split()
        lemmatized_pos_words = lemmatized_pos.split()
        changed_words = []
        for orig, lem in zip(original_words, lemmatized_pos_words):
            if orig != lem:
                changed_words.append(f"{orig}→{lem}")
        if changed_words:
            changes["LT"] = f"带词性标注的词形还原：对 {len(changed_words)} 个词进行了基于词性标注的词形还原。示例变化: {', '.join(changed_words[:5])}" + (" 等" if len(changed_words) > 5 else "")
        else:
            changes["LT"] = "带词性标注的词形还原：文本未发生变化"
    else:
        changes["LT"] = "带词性标注的词形还原：文本未发生变化"
    
    return {
        "B": base,  # Base - 原始文本
        "T": tokenized,  # Tokenization - 分词
        "N": normalized,  # Normalization - 标准化
        "R": stopwords_removed,  # Remove Stop Words - 移除停用词
        "SP": porter_stemmed,  # Porter Stemmer
        "SS": snowball_stemmed,  # Snowball Stemmer
        "SL": lancaster_stemmed,  # Lancaster Stemmer
        "L": lemmatized,  # Lemmatizer
        "LT": lemmatized_pos,  # Lemmatizer with POS
        "changes": changes  # 每个步骤的修改说明
    }

def parse_reviews_file(file_path):
    """
    解析评论文件，提取每个评论和其中的句子
    """
    reviews = []
    current_review_id = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 检查是否是新的评论开始
            if line.startswith('=== Review'):
                # 保存之前的评论
                if current_review_id is not None and current_text:
                    review_text = ' '.join(current_text)
                    reviews.append({
                        "review_id": current_review_id,
                        "review_text": review_text
                    })
                
                # 开始新评论
                match = re.search(r'Review (\d+)', line)
                if match:
                    current_review_id = f"review_{match.group(1)}"
                else:
                    current_review_id = f"review_{len(reviews) + 1}"
                current_text = []
            elif line and current_review_id:
                # 累积评论文本
                current_text.append(line)
        
        # 保存最后一个评论
        if current_review_id is not None and current_text:
            review_text = ' '.join(current_text)
            reviews.append({
                "review_id": current_review_id,
                "review_text": review_text
            })
    
    return reviews

def main():
    input_file = 'extracted_reviews_other_errors.txt'
    output_file = 'reviews_sentences_nlp_processed.json'
    
    print(f"正在读取文件: {input_file}")
    
    # 下载 NLTK 数据
    download_nltk_data()
    
    # 解析评论文件
    reviews = parse_reviews_file(input_file)
    print(f"找到 {len(reviews)} 个评论")
    
    # 处理每个评论中的句子
    all_sentences = []
    total_sentences = 0
    
    for review_idx, review in enumerate(reviews, 1):
        review_text = review["review_text"]
        review_id = review["review_id"]
        
        # 使用 NLTK 的句子分割
        sentences = sent_tokenize(review_text)
        
        for sent_idx, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            total_sentences += 1
            if total_sentences % 100 == 0:
                print(f"已处理 {total_sentences} 个句子...")
            
            # 处理句子
            processed = process_sentence(sentence)
            
            all_sentences.append({
                "sentence_id": f"{review_id}_sentence_{sent_idx}",
                "review_id": review_id,
                "sentence_index": sent_idx,
                "original": sentence,
                "processed": processed
            })
    
    print(f"总共处理了 {total_sentences} 个句子")
    
    # 保存结果
    output_data = {
        "total_reviews": len(reviews),
        "total_sentences": total_sentences,
        "preprocessing_methods": {
            "B": "Base - 原始文本",
            "T": "Tokenization - 分词",
            "N": "Normalization - 标准化",
            "R": "Remove Stop Words - 移除停用词",
            "SP": "Porter Stemmer - Porter 词干提取",
            "SS": "Snowball Stemmer - Snowball 词干提取",
            "SL": "Lancaster Stemmer - Lancaster 词干提取",
            "L": "Lemmatizer - 词形还原",
            "LT": "Lemmatizer with POS - 带词性标注的词形还原"
        },
        "note": "每个句子的 processed 字段包含所有预处理步骤的结果，以及 changes 字段说明每个步骤对原始数据的具体修改",
        "sentences": all_sentences
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n处理完成！")
    print(f"结果已保存到: {output_file}")
    print(f"总共处理了 {len(reviews)} 个评论，{total_sentences} 个句子")

if __name__ == '__main__':
    main()
