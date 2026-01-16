#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ontology-nlp 项目的 NLP 预处理代码处理评论
"""

import json
import sys
import re
import nltk

# 添加 ontology-nlp 项目路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/ontology-nlp/ontology-nlp-master')

# 尝试导入 enchant，如果失败则使用简化版本
try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False
    print("警告: enchant 库未安装，将跳过英式转美式拼写转换")

# 下载必要的 NLTK 数据
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

# 导入预处理函数（复制关键函数以避免依赖问题）
def change_to_snake_case(name):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    name = pattern.sub(' ', name)
    return name

def change_british_to_american(word):
    if not ENCHANT_AVAILABLE:
        return word  # 如果 enchant 不可用，直接返回原词
    try:
        uk_dict = enchant.Dict("en_GB")
        us_dict = enchant.Dict("en_US")
        if uk_dict.check(word) and not us_dict.check(word):
            suggestions = us_dict.suggest(word)
            return suggestions[0] if suggestions else word
    except:
        return word
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
    PORTER_STEMMER = "SP"
    SNOWBALL_STEMMER = "SS"
    LANCASTER_STEMMER = "SL"
    LEMMATIZER = "L"
    LEMMATIZER_AND_POS = "LT"
    
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
            word, tag = tagged_tokens[0]
            name = lmtzr.lemmatize(name, pos=penn2morphy(tag))
        strs.append(name)
    return ' '.join(strs)

PORTER_STEMMER = "SP"
SNOWBALL_STEMMER = "SS"
LANCASTER_STEMMER = "SL"
LEMMATIZER = "L"
LEMMATIZER_AND_POS = "LT"

def process_text(text):
    """
    对文本应用所有 NLP 预处理方法
    """
    if not text or not text.strip():
        return {}
    
    # 原始文本
    base = text.strip()
    
    # 分词
    tokenized = tokenization(base)
    
    # 标准化
    normalized = normalization(tokenized)
    
    # 移除停用词
    stopwords_removed = removing_stop_words(normalized)
    
    # 词干提取和词形还原
    porter_stemmed = get_stemming(stopwords_removed, PORTER_STEMMER)
    snowball_stemmed = get_stemming(stopwords_removed, SNOWBALL_STEMMER)
    lancaster_stemmed = get_stemming(stopwords_removed, LANCASTER_STEMMER)
    lemmatized = get_stemming(stopwords_removed, LEMMATIZER)
    lemmatized_pos = get_stemming(stopwords_removed, LEMMATIZER_AND_POS)
    
    return {
        "B": base,  # Base - 原始文本
        "T": tokenized,  # Tokenization - 分词
        "N": normalized,  # Normalization - 标准化
        "R": stopwords_removed,  # Remove Stop Words - 移除停用词
        "SP": porter_stemmed,  # Porter Stemmer
        "SS": snowball_stemmed,  # Snowball Stemmer
        "SL": lancaster_stemmed,  # Lancaster Stemmer
        "L": lemmatized,  # Lemmatizer
        "LT": lemmatized_pos  # Lemmatizer with POS
    }

def main():
    input_file = 'extracted_reviews_other_errors.txt'
    output_file = 'reviews_nlp_processed.json'
    
    print(f"正在读取文件: {input_file}")
    
    # 读取评论文件
    reviews = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 跳过空行
                reviews.append({
                    "line_number": line_num,
                    "original_text": line
                })
    
    print(f"找到 {len(reviews)} 条评论")
    print("开始处理...")
    
    # 处理每条评论
    processed_reviews = []
    for i, review in enumerate(reviews, 1):
        if i % 50 == 0:
            print(f"已处理 {i}/{len(reviews)} 条评论...")
        
        original = review["original_text"]
        processed = process_text(original)
        
        processed_reviews.append({
            "line_number": review["line_number"],
            "original_text": original,
            "processed": processed
        })
    
    # 保存结果
    output_data = {
        "total_reviews": len(processed_reviews),
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
        "reviews": processed_reviews
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n处理完成！")
    print(f"结果已保存到: {output_file}")
    print(f"总共处理了 {len(processed_reviews)} 条评论")

if __name__ == '__main__':
    main()
