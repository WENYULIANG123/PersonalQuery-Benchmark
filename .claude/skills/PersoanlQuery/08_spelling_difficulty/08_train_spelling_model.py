import os
import math
import collections
import re
import nltk
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
spelling_model = __import__("09_spelling_model")
SpellingDifficultyModel = spelling_model.SpellingDifficultyModel

spelling_features = __import__("09_spelling_features")
extract_all_features = spelling_features.extract_all_features

# Download required nltk datasets
nltk.download('words', quiet=True)
nltk.download('brown', quiet=True)
from nltk.corpus import words, brown

def generate_difficulty_label(word, freq_dict, max_freq, user_profile=None):
    score = 0.0
    word_clean = word.lower()
    
    # 1. Length factor
    length_score = min(len(word_clean) / 15.0, 1.0) * 0.25
    score += length_score
    
    # 2. Frequency factor
    freq = freq_dict.get(word_clean, 1)
    if max_freq > 0:
        freq_score = (1.0 - (math.log(freq + 1) / math.log(max_freq + 1))) * 0.25
    else:
        freq_score = 0.25
    score += freq_score
    
    # 3. Linguistic traps
    trap_score = 0.0
    complex_vowels = ['ough', 'augh', 'eigh', 'ieu', 'oe', 'ae']
    for v in complex_vowels:
        if v in word_clean: trap_score += 0.2
        
    splits = re.split(r'[aeiouy]+', word_clean)
    max_consonants = max(len(c) for c in splits) if splits else 0
    if max_consonants >= 3: trap_score += 0.15
    if max_consonants >= 4: trap_score += 0.2
    
    silent_starts = ['kn', 'wr', 'ps', 'pn', 'gh']
    if any(word_clean.startswith(p) for p in silent_starts): trap_score += 0.2
    
    trap_score = min(trap_score, 0.5)
    score += trap_score
    
    if user_profile is not None:
        hw, dele, homo, suff, esp, trans, ins, sub, scr = user_profile
        
        homophones = {'there', 'their', 'theyre', 'its', 'your', 'youre', 'to', 'too', 'affect', 'effect', 'accept', 'except', 'bare', 'bear', 'break', 'brake', 'buy', 'by', 'principal', 'principle', 'right', 'write', 'than', 'then', 'weather', 'whether', 'whose'}
        if word_clean in homophones:
            score += homo * 0.45
            
        suffixes = ('s', 'es', 'ed', 'ing', 'ly', 'tion', 'sion', 'ment', 'able', 'ible', 'ness', 'ful', 'less')
        if word_clean.endswith(suffixes) and len(word_clean) >= 4:
            score += suff * 0.35
            
        has_double = any(word_clean[i] == word_clean[i+1] for i in range(len(word_clean)-1))
        if has_double:
            score += dele * 0.3
            score += ins * 0.3
        elif len(word_clean) > 7:
            score += dele * 0.2
            score += scr * 0.2
            
        if any(pair in word_clean for pair in ['ie', 'ei', 'ou', 'uo', 'ea', 'ae', 'ai', 'ia']):
            score += trans * 0.35
            
        score += hw * (score * 0.6)
    
    return min(max(0.0, score), 1.0)

def prepare_data():
    print("Loading vocabulary and frequencies...")
    all_words = list(set([w.lower() for w in words.words() if w.isalpha() and len(w) > 2]))
    # limit to 20,000 words to make it faster for training
    all_words = all_words[:20000]
    
    freq_dict = collections.Counter(w.lower() for w in brown.words() if w.isalpha())
    max_freq = max(freq_dict.values()) if freq_dict else 1
    
    char2idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    
    print(f"Preparing features for {len(all_words)} words...")
    X_chars, X_feats, X_user, Y_labels = [], [], [], []
    for w in all_words:
        idx_list = [char2idx.get(c, 0) for c in w]
        if len(idx_list) > 32:
            idx_list = idx_list[:32]
        else:
            idx_list += [0] * (32 - len(idx_list))
            
        feats = extract_all_features(w, freq_dict, max_freq)
        
        # generate 2 random user profiles per word
        for _ in range(2):
            raw_prof = np.random.rand(9)
            user_prof = list(raw_prof / raw_prof.sum())
            label = generate_difficulty_label(w, freq_dict, max_freq, user_prof)
            
            X_chars.append(idx_list)
            X_feats.append(feats)
            X_user.append(user_prof)
            Y_labels.append([label])

    X_chars = torch.tensor(X_chars, dtype=torch.long)
    X_feats = torch.tensor(X_feats, dtype=torch.float32)
    X_user = torch.tensor(X_user, dtype=torch.float32)
    Y_labels = torch.tensor(Y_labels, dtype=torch.float32)
    
    return TensorDataset(X_chars, X_feats, X_user, Y_labels)

def train_model():
    dataset = prepare_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = SpellingDifficultyModel(vocab_size=28, num_handcrafted_features=50, num_user_features=9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    epochs = 15

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for chars, feats, user_feats, labels in train_loader:
            chars, feats, user_feats, labels = chars.to(device), feats.to(device), user_feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(chars, feats, user_feats)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for chars, feats, user_feats, labels in val_loader:
                chars, feats, user_feats, labels = chars.to(device), feats.to(device), user_feats.to(device), labels.to(device)
                val_preds = model(chars, feats, user_feats)
                val_loss += criterion(val_preds, labels).item()
                
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), '08_spelling_difficulty_scorer_v1.pt')
            print(f"  Saved best model with Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_model()
