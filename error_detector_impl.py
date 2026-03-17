"""
完全独立实现的英文错误检测系统
独立于AllenNLP，使用纯Transformers实现
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import warnings
warnings.filterwarnings('ignore')


class ErrorDetectionModel:
    """序列标注模型用于错误检测"""
    
    ERROR_TYPES = {
        'SPELLING': 'Spelling/Typo errors',
        'GRAMMAR': 'Grammar errors (tense, agreement, etc)',
        'PUNCTUATION': 'Punctuation errors',
        'WORD_CHOICE': 'Word choice/vocabulary',
        'DELETION': 'Unnecessary word deletion',
        'MERGE': 'Word merge/spacing',
    }
    
    def __init__(self, model_name: str = 'roberta-base', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=2
        ).to(self.device)
        self.model.eval()
    
    def detect_errors(self, sentences: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        检测句子中的错误
        
        Args:
            sentences: 待检测的句子列表
            threshold: 错误概率阈值 (0-1)
        
        Returns:
            检测结果列表
        """
        results = []
        
        with torch.no_grad():
            for sent_idx, sentence in enumerate(sentences):
                tokens = sentence.split()
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**encoding)
                logits = outputs.logits[0].cpu().numpy()
                
                probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                error_class_probs = probabilities[:, 1]
                
                errors = []
                word_ids = encoding.word_ids()
                
                for token_idx, word_id in enumerate(word_ids):
                    if word_id is None or word_id >= len(tokens):
                        continue
                    
                    prob = float(error_class_probs[token_idx])
                    
                    if prob >= threshold:
                        error_type = self._classify_error_type(tokens[word_id])
                        errors.append({
                            'position': word_id,
                            'token': tokens[word_id],
                            'error_type': error_type,
                            'confidence': prob
                        })
                
                unique_errors_by_pos = {e['position']: e for e in errors}
                errors = sorted(unique_errors_by_pos.values(), key=lambda x: x['position'])
                
                results.append({
                    'sent_idx': sent_idx,
                    'sentence': sentence,
                    'tokens': tokens,
                    'num_errors': len(errors),
                    'errors': errors
                })
        
        return results
    
    @staticmethod
    def _classify_error_type(token: str) -> str:
        """简单的错误分类启发式"""
        token_lower = token.lower()
        
        common_misspellings = {
            'teh': 'the', 'recieved': 'received', 'occured': 'occurred',
            'begining': 'beginning', 'excelent': 'excellent', 'seperate': 'separate'
        }
        if token_lower in common_misspellings:
            return 'SPELLING'
        
        if token.endswith(('ing', 'ed', 'es')) and len(token) > 3:
            return 'GRAMMAR'
        
        if token in ',.!?;:':
            return 'PUNCTUATION'
        
        return 'WORD_CHOICE'


class EnglishErrorDetector:
    """完整的错误检测系统，包括规则引擎和模型"""
    
    def __init__(self, use_model: bool = False, model_name: str = 'roberta-base'):
        self.use_model = use_model
        if use_model:
            self.model = ErrorDetectionModel(model_name)
        else:
            self.model = None
        
        self.rule_engine = RuleBasedDetector()
    
    def detect(self, sentences: List[str], threshold: float = 0.5) -> List[Dict]:
        """检测句子中的所有错误"""
        
        if self.use_model and self.model is not None:
            return self.model.detect_errors(sentences, threshold)
        else:
            return self.rule_engine.detect(sentences)


class RuleBasedDetector:
    """基于规则的错误检测器 (不需要模型权重)"""
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> Dict:
        return {
            'SPELLING': [
                ('teh', 'the'),
                ('recieved', 'received'),
                ('occured', 'occurred'),
                ('begining', 'beginning'),
                ('excelent', 'excellent'),
                ('seperate', 'separate'),
                ('occassion', 'occasion'),
                ('untill', 'until'),
                ('alot', 'a lot'),
            ],
            'GRAMMAR': [
                ('is bad$', 'is bad'),  # 要用"is"
                ('likes$', 'should be "like"'),  # I + like
                ('doesnt', "doesn't"),
                ('thier', 'their'),
                ('its a', "it's a"),
            ],
            'PUNCTUATION': [
                ('quality good', 'quality, good'),  # 缺逗号
            ]
        }
    
    def detect(self, sentences: List[str]) -> List[Dict]:
        results = []
        
        for sent_idx, sentence in enumerate(sentences):
            tokens = sentence.split()
            errors = []
            
            for word_idx, word in enumerate(tokens):
                word_lower = word.lower().rstrip('.,!?;:')
                
                for error_word, correction in self.rules['SPELLING']:
                    if word_lower == error_word:
                        errors.append({
                            'position': word_idx,
                            'token': word,
                            'error_type': 'SPELLING',
                            'correction': correction,
                            'confidence': 0.95
                        })
                        break
                
                if word_lower in ('likes', 'wants', 'needs') and word_idx > 0:
                    if tokens[word_idx - 1].lower() == 'i':
                        singular_form = word[:-1]
                        errors.append({
                            'position': word_idx,
                            'token': word,
                            'error_type': 'GRAMMAR',
                            'correction': singular_form,
                            'confidence': 0.92
                        })
            
            combined_text = ' '.join(tokens).lower()
            if 'quality good' in combined_text:
                errors.append({
                    'position': -1,
                    'token': 'quality good',
                    'error_type': 'PUNCTUATION',
                    'issue': 'Missing comma between adjectives',
                    'confidence': 0.85
                })
            
            results.append({
                'sent_idx': sent_idx,
                'sentence': sentence,
                'tokens': tokens,
                'num_errors': len(errors),
                'errors': sorted(errors, key=lambda x: x['position'])
            })
        
        return results


def print_detection_results(results: List[Dict], show_corrections: bool = True):
    """美化打印检测结果"""
    print("\n" + "="*90)
    print("ERROR DETECTION RESULTS")
    print("="*90)
    
    total_errors = sum(r['num_errors'] for r in results)
    
    for result in results:
        sent = result['sentence']
        num_errors = result['num_errors']
        
        status = "✓ OK" if num_errors == 0 else f"⚠ {num_errors} error(s)"
        print(f"\n📝 {sent}")
        print(f"   Status: {status}")
        
        if result['errors']:
            print(f"\n   {'Pos':<4} {'Token':<20} {'Type':<15} {'Info':<40} {'Conf.':<6}")
            print(f"   {'-'*85}")
            
            for err in result['errors']:
                pos = err.get('position', '?')
                token = err['token'][:19]
                err_type = err['error_type']
                
                if 'correction' in err:
                    info = f"→ {err['correction']}"[:39]
                elif 'issue' in err:
                    info = err['issue'][:39]
                else:
                    info = "Fix needed"
                
                conf = f"{err['confidence']:.0%}"
                print(f"   {pos:<4} {token:<20} {err_type:<15} {info:<40} {conf:<6}")
    
    print("\n" + "="*90)
    print(f"SUMMARY: {len(results)} sentences, {total_errors} errors detected")
    print("="*90 + "\n")


if __name__ == '__main__':
    test_sentences = [
        "This product is excellent with great quality.",
        "Teh product quality is really good.",
        "I likes this product very much.",
        "Great quality good price excellent service.",
        "I recieved the package very quickly.",
        "The delivery is very fast but product quality are bad.",
    ]
    
    print("🔧 初始化错误检测器...")
    detector = EnglishErrorDetector(use_model=False)  # 使用规则引擎
    
    print(f"📊 处理 {len(test_sentences)} 条句子...\n")
    results = detector.detect(test_sentences, threshold=0.5)
    
    print_detection_results(results)
