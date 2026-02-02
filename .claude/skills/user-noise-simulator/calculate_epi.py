#!/usr/bin/env python3
"""
EPI (Error Proneness Index) Calculation Script - Enhanced Edition
Supports 14-category error style recommendations based on User DNA and word features.
"""

import os
import json
import sys
import re
import math

# Define the 14 categories for reference
CATEGORIES = {
    1: "Deletion", 2: "Insertion", 3: "Transposition", 4: "Substitution",
    5: "Homophone", 6: "Hard Word", 7: "Extra Space", 8: "Extra Hyphen",
    9: "Suffix", 10: "Agreement", 11: "Hyphenation", 12: "Pronoun",
    13: "Collocation", 14: "Preposition"
}

def assess_word_difficulty(word):
    """Assess word difficulty based on scientific factors."""
    word = word.lower().strip()
    L_w = len(word)
    
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'with', 'hair', 'skin', 'good', 'like', 'love', 'best', 'make', 'free', 'natural', 'water', 'daily', 'great', 'nice', 'soft', 'fine', 'full', 'long', 'high', 'low', 'deep', 'light', 'dark', 'hot', 'cold', 'looking', 'shampoo', 'conditioner'}
    technical_beauty = {'hyaluronic', 'moisturizer', 'sulfate', 'keratin', 'peptide', 'vitamin', 'protein', 'antioxidant', 'regimen', 'emollient', 'humectant', 'occlusive', 'surfactant', 'curly', 'frizz', 'volume', 'thickness', 'weighing', 'argan', 'define', 'reduce'}

    if word in common_words:
        F_w = 2.0
    elif word in technical_beauty:
        F_w = 12.0
    else:
        F_w = 6.0

    vowels = 'aeiou'
    syllable_count = sum(1 for char in word if char in vowels)
    consonant_clusters = sum(1 for i in range(len(word)-1) if word[i] not in vowels and word[i+1] not in vowels)
    P_w = (syllable_count * 0.6) + (consonant_clusters * 0.4)
    D_w = (L_w * F_w * P_w) / 100.0

    return {
        'difficulty_score': round(D_w, 2),
        'difficulty_level': 'low' if D_w < 3.0 else 'medium' if D_w < 6.0 else 'high',
        'length': L_w,
        'phonetic_complexity': round(P_w, 2)
    }

def assess_user_dna(dna_file):
    """Load and assess user's error DNA from style_analysis JSON."""
    if not os.path.exists(dna_file):
        return None
        
    with open(dna_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data.get('results'):
        return None
        
    stats = data['results'][0].get('statistics', {})
    total_errors = stats.get('total_errors', 0)
    total_words = stats.get('total_words', 1000)
    base_error_rate = (total_errors / total_words) * 100
    
    t_u = {}
    for dim_key in ['dimension_1_character_word_variants', 'dimension_2_grammar_structural']:
        by_cat = stats.get(dim_key, {}).get('by_category', {})
        for cat, count in by_cat.items():
            t_u[cat] = count / total_errors if total_errors > 0 else 0
            
    return {
        'base_error_rate': base_error_rate,
        'error_type_vector': t_u,
        'user_id': data['results'][0].get('user_id', 'Unknown')
    }

def recommend_error_styles(word, difficulty, user_dna):
    """Recommend error styles based on word features and User DNA."""
    scores = {}
    word_lower = word.lower()
    
    # 1. Base scores from User DNA
    for cat_id, cat_name in CATEGORIES.items():
        scores[cat_name] = user_dna['error_type_vector'].get(cat_name, 0.01) # Default tiny prob

    # 2. Dynamic Adjustments based on Word Features
    L = difficulty['length']
    D = difficulty['difficulty_score']
    
    # Feature 1: Word Length influence
    if L > 8:
        scores['Extra Space'] *= 2.0
        scores['Transposition'] *= 1.5
        scores['Hard Word'] *= 1.8
        scores['Deletion'] *= 1.2
    
    # Feature 2: Phonetic patterns
    phonetic_triggers = ['tion', 'sion', 'ough', 'eigh', 'ph', 'ch', 'que']
    if any(t in word_lower for t in phonetic_triggers):
        scores['Homophone'] *= 2.5
        scores['Substitution'] *= 1.5

    # Feature 3: Suffixes
    suffix_triggers = ['ing', 'ed', 'ly', 'ment', 'able', 'ness']
    if any(word_lower.endswith(s) for s in suffix_triggers):
        scores['Suffix'] *= 3.0

    # Feature 4: Compound/Potential space
    if L > 10:
        scores['Extra Space'] *= 1.5
        scores['Extra Hyphen'] *= 1.2

    # Feature 5: Simple words (low difficulty)
    if D < 3.0:
        scores['Deletion'] *= 2.0
        scores['Insertion'] *= 1.8
        scores['Substitution'] *= 1.5
        scores['Homophone'] *= 0.5
        scores['Hard Word'] *= 0.2

    # 3. Sort and Rank
    sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Normalize for display
    total_score = sum(scores.values())
    recommendations = []
    for name, score in sorted_recommendations[:3]:
        confidence = (score / total_score) * 100 if total_score > 0 else 0
        recommendations.append({
            'style': name,
            'confidence': round(confidence, 1)
        })
        
    return recommendations

def calculate_epi(word, user_dna):
    """Calculate Error Proneness Index (EPI)."""
    difficulty = assess_word_difficulty(word)
    d_w = difficulty['difficulty_score']
    
    l_w = min(len(word) / 15, 1.0)
    vowels = 'aeiou'
    consonant_clusters = sum(1 for i in range(len(word)-2) 
                           if word[i].lower() not in vowels and word[i+1].lower() not in vowels and word[i+2].lower() not in vowels)
    c_w = min(consonant_clusters * 0.3, 1.0)
    
    e_u = user_dna['base_error_rate'] / 100.0
    epi = (d_w * e_u) * (l_w * 0.6 + c_w * 0.4)
    epi_normalized = 1 / (1 + 2.71828 ** (-epi * 2))
    
    recommendations = recommend_error_styles(word, difficulty, user_dna)
    
    return {
        'word': word,
        'epi': round(epi_normalized, 4),
        'difficulty': difficulty,
        'recommendations': recommendations
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 calculate_epi.py <query_text> <dna_json_path>")
        sys.exit(1)
        
    query = sys.argv[1]
    dna_path = sys.argv[2]
    
    user_dna = assess_user_dna(dna_path)
    if not user_dna:
        user_dna = {'base_error_rate': 1.0, 'error_type_vector': {}, 'user_id': 'Default'}
        
    words = re.findall(r'\b\w+\b', query)
    results = []
    for word in words:
        if len(word) < 3: continue
        results.append(calculate_epi(word, user_dna))
        
    results.sort(key=lambda x: x['epi'], reverse=True)
    
    print(f"\n[EPI Analysis for User: {user_dna['user_id']}]")
    print(f"{'Word':<15} | {'EPI':<8} | {'Diff':<6} | {'Top Recommended Style (14 Cats)'}")
    print("-" * 75)
    for r in results[:5]:
        rec_str = f"{r['recommendations'][0]['style']} ({r['recommendations'][0]['confidence']}%)"
        print(f"{r['word']:<15} | {r['epi']:<8} | {r['difficulty']['difficulty_level']:<6} | {rec_str}")
        
    print("\n[Deep Reasoning for Injection]")
    if results:
        top = results[0]
        print(f"🎯 Target: '{top['word']}'")
        print(f"💡 Why: It has the highest EPI ({top['epi']}).")
        print(f"🔧 How: Inject '{top['recommendations'][0]['style']}' (Confidence: {top['recommendations'][0]['confidence']}%).")
        alt_list = [f"{r['style']} ({r['confidence']}%)" for r in top['recommendations'][1:]]
        print(f"👉 Alternatives: {', '.join(alt_list)}")

if __name__ == "__main__":
    main()
