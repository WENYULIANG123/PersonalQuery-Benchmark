#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any
from pathlib import Path


class GectorDetectorHF:
    def __init__(self,
                 model_id: str = "sahilnishad/BERT-GED-FCE-FT",
                 min_error_probability: float = 0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_error_probability = min_error_probability
        self.model_id = model_id
        
        print(f"Device: {self.device}")
        print(f"Loading tokenizer from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print(f"Loading token classification model from: {model_id}")
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = {0: "CORRECT", 1: "INCORRECT"}
        self.label2id = {"CORRECT": 0, "INCORRECT": 1}
        self.num_labels = self.model.config.num_labels
        
        print(f"Model loaded. Binary classification: {self.id2label}")
    
    def detect_errors(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        results = []
        
        for text in texts:
            tokens = text.split()
            errors = self._detect_in_text(text, tokens)
            
            results.append({
                "text": text,
                "errors": errors,
                "error_count": len(errors)
            })
        
        return results
    
    def _detect_in_text(self, text: str, tokens: List[str]) -> List[Dict[str, Any]]:
        encodings = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        
        word_ids = encodings.word_ids()
        errors = []
        
        for word_idx, token in enumerate(tokens):
            token_mask = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if token_mask:
                token_logits = logits[0, token_mask[0], :]
                token_probs = F.softmax(token_logits, dim=-1)
                
                incorrect_prob = token_probs[1].item()
                correct_prob = token_probs[0].item()
                
                if incorrect_prob >= self.min_error_probability:
                    errors.append({
                        "token_idx": word_idx,
                        "token": token,
                        "prediction": "INCORRECT",
                        "incorrect_probability": float(incorrect_prob),
                        "correct_probability": float(correct_prob),
                        "error_type": "GRAMMAR"
                    })
        
        return errors

