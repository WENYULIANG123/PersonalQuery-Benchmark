import os
import torch
import torch.nn as nn
import json
import collections
import re
spelling_model = __import__("08_spelling_model")
SpellingDifficultyModel = spelling_model.SpellingDifficultyModel

spelling_features = __import__("08_spelling_features")
extract_all_features = spelling_features.extract_all_features

# To prevent dependency issues, we can just load the frequency dictionary once.
# In a real deployed version, we could cache this dict. For now, we load on init.
import nltk
from nltk.corpus import brown

class SpellingDifficultyScorer:
    def __init__(self, model_path=None):
        nltk.download('brown', quiet=True)
        self.freq_dict = collections.Counter(w.lower() for w in brown.words() if w.isalpha())
        self.max_freq = max(self.freq_dict.values()) if self.freq_dict else 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpellingDifficultyModel(vocab_size=28, num_handcrafted_features=50, num_user_features=9)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.char2idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}

    def predict_difficulty(self, word, user_profile=None):
        word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(word_clean) < 3:
            return 0.0
            
        idx_list = [self.char2idx.get(c, 0) for c in word_clean]
        if len(idx_list) > 32:
            idx_list = idx_list[:32]
        else:
            idx_list += [0] * (32 - len(idx_list))
            
        feats = extract_all_features(word_clean, self.freq_dict, self.max_freq)
        
        X_chars = torch.tensor([idx_list], dtype=torch.long).to(self.device)
        X_feats = torch.tensor([feats], dtype=torch.float32).to(self.device)
        
        user_features = [0.0] * 9
        if user_profile and 'spelling' in user_profile:
            spelling_stats = user_profile['spelling']
            spelling_total = user_profile.get('spelling_total', sum(spelling_stats.values()))
            if spelling_total > 0:
                mapping = ['Hard Word', 'Deletion', 'Homophone', 'Suffix', 'Extra Space', 'Transposition', 'Insertion', 'Substitution', 'Scramble']
                for i, k in enumerate(mapping):
                    user_features[i] = spelling_stats.get(k, 0.0) / spelling_total
                    
        X_user = torch.tensor([user_features], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            score = self.model(X_chars, X_feats, X_user)
            
        base_score = score.item()
        return base_score

    def find_vulnerable_words(self, query: str, threshold: float = 0.4, user_profile: dict = None) -> list:
        words = query.split()
        vulnerable = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) > 3:
                diff_score = self.predict_difficulty(clean_word, user_profile=user_profile)
                if diff_score > threshold:
                    vulnerable.append((clean_word, diff_score))
        return sorted(vulnerable, key=lambda x: x[1], reverse=True)
