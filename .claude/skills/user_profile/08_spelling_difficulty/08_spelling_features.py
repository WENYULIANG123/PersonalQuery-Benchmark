import math
import collections
import re

def extract_char_features(word):
    word_lower = word.lower()
    features = {}
    
    features['length'] = len(word)
    features['num_unique_chars'] = len(set(word_lower))
    features['char_diversity'] = len(set(word_lower)) / max(len(word), 1)
    
    vowels = set('aeiou')
    features['num_vowels'] = sum(1 for c in word_lower if c in vowels)
    features['num_consonants'] = sum(1 for c in word_lower if c.isalpha() and c not in vowels)
    features['vowel_consonant_ratio'] = features['num_vowels'] / max(features['num_consonants'], 1)
    
    max_consonant_cluster = 0
    current_cluster = 0
    for c in word_lower:
        if c not in vowels and c.isalpha():
            current_cluster += 1
            max_consonant_cluster = max(max_consonant_cluster, current_cluster)
        else:
            current_cluster = 0
    features['max_consonant_cluster'] = max_consonant_cluster
    features['has_difficult_clusters'] = 1.0 if max_consonant_cluster >= 3 else 0.0
    
    char_counts = collections.Counter(word_lower)
    total = len(word_lower)
    entropy = -sum((count/total) * math.log2(count/total) for count in char_counts.values()) if total > 0 else 0
    features['char_entropy'] = entropy
    
    features['num_repeated_pairs'] = sum(1 for i in range(len(word_lower)-1) if word_lower[i] == word_lower[i+1])
    features['has_consecutive_repeats'] = 1.0 if features['num_repeated_pairs'] > 0 else 0.0
    
    uncommon = 'zqxjkwv'
    features['num_uncommon_letters'] = sum(1 for c in word_lower if c in uncommon)
    features['has_uncommon_letters'] = 1.0 if features['num_uncommon_letters'] > 0 else 0.0
    
    silent_patterns = ['gh', 'kn', 'wr', 'mb', 'bt', 'mn']
    features['num_silent_patterns'] = sum(1 for p in silent_patterns if p in word_lower)
    
    return features

def extract_pronunciation_features(word):
    features = {}
    word_lower = word.lower()
    
    long_vowel_patterns = ['ee', 'ea', 'ai', 'ay', 'oa', 'ow', 'igh']
    short_vowel_patterns = ['a', 'e', 'i', 'o', 'u']
    
    features['num_long_vowel_patterns'] = sum(1 for p in long_vowel_patterns if p in word_lower)
    features['num_short_vowel_patterns'] = sum(1 for c in short_vowel_patterns if c in word_lower)
    features['has_irregular_vowels'] = 1.0 if features['num_long_vowel_patterns'] > 0 else 0.0
    
    features['has_silent_e'] = 1.0 if word_lower.endswith('e') else 0.0
    
    digraphs = ['sh', 'ch', 'th', 'ng', 'ph', 'wh', 'gh', 'ck']
    features['num_digraphs'] = sum(1 for d in digraphs if d in word_lower)
    
    features['has_soft_c'] = 1.0 if any(word_lower[i] in 'eiy' for i in range(len(word_lower)-1) if word_lower[i] == 'c') else 0.0
    features['has_hard_c'] = 1.0 if any(word_lower[i] in 'aou' for i in range(len(word_lower)-1) if word_lower[i] == 'c') else 0.0
    
    features['has_soft_g'] = 1.0 if any(word_lower[i] in 'eiy' for i in range(len(word_lower)-1) if word_lower[i] == 'g') else 0.0
    features['has_hard_g'] = 1.0 if any(word_lower[i] in 'aou' for i in range(len(word_lower)-1) if word_lower[i] == 'g') else 0.0
    
    return features

def extract_visual_similarity_features(word):
    word_lower = word.lower()
    features = {}
    
    confusing_pairs = ['rn', 'vv', 'cl', 'nn']
    for p in confusing_pairs:
        features[f'has_{p}'] = 1.0 if p in word_lower else 0.0
        
    reversed_word = word_lower[::-1]
    features['is_palindrome'] = 1.0 if word_lower == reversed_word and len(word_lower)>1 else 0.0
    
    common_misspellings = ['ie', 'ei', 'ieve', 'eave']
    for p in common_misspellings:
        features[f'has_misspell_pattern_{p}'] = 1.0 if p in word_lower else 0.0
    
    return features

def extract_positional_features(word):
    word_lower = word.lower()
    features = {}
    
    features['starts_with_vowel'] = 1.0 if len(word_lower)>0 and word_lower[0] in 'aeiou' else 0.0
    features['starts_with_consonant_cluster'] = 1.0 if len(word_lower) > 1 and (word_lower[0] not in 'aeiou' and word_lower[1] not in 'aeiou') else 0.0
    
    features['ends_with_silent_e'] = 1.0 if word_lower.endswith('e') else 0.0
    features['ends_with_ck'] = 1.0 if word_lower.endswith('ck') else 0.0
    features['ends_with_ous'] = 1.0 if word_lower.endswith('ous') else 0.0
    features['ends_with_tion'] = 1.0 if word_lower.endswith('tion') else 0.0
    features['ends_with_sion'] = 1.0 if word_lower.endswith('sion') else 0.0
    
    features['q_without_u'] = 1.0 if 'q' in word_lower and 'qu' not in word_lower else 0.0
    
    features['y_as_vowel_at_end'] = 1.0 if word_lower.endswith('y') else 0.0
    features['y_as_vowel_in_middle'] = 0.0
    if 'y' in word_lower:
        for i in range(1, len(word_lower)-1):
            if word_lower[i] == 'y':
                if (word_lower[i-1] not in 'aeiou' and word_lower[i+1] not in 'aeiou'):
                    features['y_as_vowel_in_middle'] = 1.0
                    break
    
    return features

def extract_frequency_features(word, freq_dict, max_freq):
    features = {}
    freq = freq_dict.get(word.lower(), 1)
    features['log_frequency'] = math.log(freq + 1)
    
    features['frequency_percentile'] = freq / max_freq if max_freq > 0 else 0
    return features

def extract_all_features(word, freq_dict, max_freq):
    f1 = extract_char_features(word)
    f2 = extract_pronunciation_features(word)
    f3 = extract_visual_similarity_features(word)
    f4 = extract_positional_features(word)
    f5 = extract_frequency_features(word, freq_dict, max_freq)
    
    all_f = {**f1, **f2, **f3, **f4, **f5}
    # ensure fixed size (50) and fixed order mapping
    keys = sorted(list(all_f.keys()))
    val_list = [all_f[k] for k in keys]
    # pad to 50
    if len(val_list) < 50:
        val_list += [0.0] * (50 - len(val_list))
    return val_list[:50]
