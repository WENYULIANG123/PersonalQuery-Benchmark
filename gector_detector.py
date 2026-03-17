#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, RobertaModel
from typing import List, Dict, Any
from pathlib import Path
import json


class GectorDetector:
    def __init__(self,
                 vocab_dir: str = "/fs04/ar57/wenyu/gector/data/output_vocabulary",
                 min_error_probability: float = 0.1,
                 model_name: str = "roberta-base"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_error_probability = min_error_probability
        
        print(f"Device: {self.device}")
        print(f"Loading RoBERTa tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading RoBERTa model: {model_name}")
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.labels = self._load_vocab(vocab_dir, "labels.txt")
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        
        hidden_size = self.encoder.config.hidden_size
        num_labels = len(self.labels)
        
        self.classifier = torch.nn.Linear(hidden_size, num_labels).to(self.device)
        self.classifier.eval()
    
    def _load_vocab(self, vocab_dir: str, filename: str) -> List[str]:
        filepath = Path(vocab_dir) / filename
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
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
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            
            logits = self.classifier(last_hidden)
        
        probs = F.softmax(logits, dim=-1)
        
        word_ids = encodings.word_ids()
        errors = []
        
        seen_token_ids = set()
        for word_idx, token in enumerate(tokens):
            token_mask = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if token_mask:
                token_logits = logits[0, token_mask[0], :]
                token_probs = F.softmax(token_logits, dim=-1)
                
                max_prob, pred_label_idx = torch.max(token_probs, dim=0)
                max_prob = max_prob.item()
                pred_label = self.id2label.get(pred_label_idx.item(), "$KEEP")
                
                if max_prob >= self.min_error_probability and pred_label != "$KEEP":
                    errors.append({
                        "token_idx": word_idx,
                        "token": token,
                        "predicted_operation": pred_label,
                        "confidence": float(max_prob),
                        "error_type": self._classify_error_type(pred_label)
                    })
        
        return errors
    
    def _classify_error_type(self, operation: str) -> str:
        if operation.startswith("$REPLACE") or operation.startswith("$APPEND"):
            if operation in ["$REPLACE_.", "$APPEND_.", "$REPLACE_,", "$APPEND_,", 
                             "$REPLACE_;", "$APPEND_;", "$REPLACE_:", "$APPEND_:"]:
                return "PUNCTUATION"
            else:
                return "GRAMMAR"
        elif operation == "$DELETE":
            return "GRAMMAR"
        elif operation.startswith("$TRANSFORM"):
            if any(x in operation for x in ["VERB", "AGREEMENT", "CASE", "TENSE"]):
                return "GRAMMAR"
            else:
                return "SPELLING"
        else:
            return "UNKNOWN"


def load_json_comments(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [item.get("text", item) if isinstance(item, dict) else str(item) for item in data]
    elif isinstance(data, dict):
        return [data.get("text", data)]
    else:
        return []


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gector_detector.py <input.json> [output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "gector_results.json"
    
    print(f"Loading comments from {input_file}...")
    texts = load_json_comments(input_file)
    
    print(f"Initializing GECToR detector...")
    detector = GectorDetector()
    
    print(f"Processing {len(texts)} comments...")
    results = detector.detect_errors(texts)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
