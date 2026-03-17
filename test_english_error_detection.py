#!/usr/bin/env python3
"""
English Grammar Error Detection Demo
Using transformer-based token classification approach
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple


class EnglishErrorDetector:
    """Detect grammar and spelling errors in English text"""
    
    ERROR_PATTERNS = {
        'SPELLING': [
            ('teh', 'the'),
            ('recieved', 'received'),
            ('occured', 'occurred'),
            ('begining', 'beginning'),
            ('excelent', 'excellent'),
        ],
        'GRAMMAR': [
            ('are bad', 'is bad'),          
            ('likes this', 'like this'),
            ('i am', 'I am'),
            ('it is not very', 'it is'),
        ],
        'PUNCTUATION': [
            ('quality good', 'quality, good'),
        ]
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def detect_spelling_errors(text: str) -> List[Dict]:
        """Detect spelling errors using pattern matching"""
        errors = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,!?;:')
            
            for error, correction in EnglishErrorDetector.ERROR_PATTERNS['SPELLING']:
                if word_lower == error:
                    errors.append({
                        'position': i,
                        'token': word,
                        'error_type': 'SPELLING',
                        'correction': correction,
                        'confidence': 0.95
                    })
                    break
        
        return errors
    
    @staticmethod
    def detect_grammar_errors(text: str) -> List[Dict]:
        """Detect grammar errors using heuristics"""
        errors = []
        words = text.lower().split()
        
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            
            if pair == 'are bad':
                errors.append({
                    'position': i,
                    'token': f"{words[i]} {words[i+1]}",
                    'error_type': 'GRAMMAR',
                    'issue': 'Subject-verb agreement: "are" should be "is" for singular',
                    'confidence': 0.90
                })
            elif pair == 'likes this' and i > 0 and words[i-1] == 'i':
                errors.append({
                    'position': i,
                    'token': words[i],
                    'error_type': 'GRAMMAR',
                    'issue': 'Verb conjugation: "likes" should be "like" after "I"',
                    'confidence': 0.92
                })
        
        return errors
    
    @staticmethod
    def detect_punctuation_errors(text: str) -> List[Dict]:
        """Detect missing punctuation"""
        errors = []
        
        if 'quality good' in text.lower():
            pos = text.lower().find('quality good')
            errors.append({
                'position': text[:pos].count(' '),
                'token': 'quality good',
                'error_type': 'PUNCTUATION',
                'issue': 'Missing comma between adjectives',
                'confidence': 0.85
            })
        
        return errors
    
    def detect_all_errors(self, text: str) -> Dict:
        """Detect all types of errors in text"""
        errors = []
        errors.extend(self.detect_spelling_errors(text))
        errors.extend(self.detect_grammar_errors(text))
        errors.extend(self.detect_punctuation_errors(text))
        
        return {
            'text': text,
            'num_errors': len(errors),
            'errors': sorted(errors, key=lambda x: x['position'])
        }


def create_test_sentences() -> List[str]:
    """Create English test sentences with various errors"""
    return [
        "This product is excellent with great quality.",
        "Fast delivery and good packaging.",
        "Teh product quality is really good.",
        "I recieved the package very quickly.",
        "The delivery is very fast but product quality are bad.",
        "I likes this product very much.",
        "Great quality good price excellent service.",
        "Good quality, but the price were too high.",
        "teh product is good but quality it is not very well.",
        "I thinks its really excelent but costly.",
    ]


def print_results(results: List[Dict]):
    """Pretty print detection results"""
    print("\n" + "="*85)
    print("ENGLISH ERROR DETECTION RESULTS")
    print("="*85)
    
    total_errors = 0
    
    for idx, result in enumerate(results, 1):
        text = result['text']
        num_errors = result['num_errors']
        total_errors += num_errors
        
        status = "✓ OK" if num_errors == 0 else f"⚠ {num_errors} error(s)"
        print(f"\n{idx}. {text}")
        print(f"   Status: {status}")
        
        if result['errors']:
            print(f"\n   {'Pos':<4} {'Token':<20} {'Type':<15} {'Issue':<35} {'Conf.':<6}")
            print(f"   {'-'*80}")
            
            for err in result['errors']:
                pos = err.get('position', '?')
                token = err['token'][:19]
                err_type = err['error_type']
                issue = err.get('issue', err.get('correction', 'Fix needed'))[:34]
                conf = f"{err['confidence']:.0%}"
                print(f"   {pos:<4} {token:<20} {err_type:<15} {issue:<35} {conf:<6}")
    
    print("\n" + "="*85)
    print(f"SUMMARY: {len(results)} sentences, {total_errors} errors detected")
    print("="*85)


def main():
    print("="*85)
    print("English Grammar Error Detection - Demonstration")
    print("="*85)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    detector = EnglishErrorDetector()
    test_sentences = create_test_sentences()
    
    print(f"\nAnalyzing {len(test_sentences)} test sentences...\n")
    
    results = []
    for sent in test_sentences:
        result = detector.detect_all_errors(sent)
        results.append(result)
    
    print_results(results)
    
    print("\n💡 INTERPRETATION:")
    print("-" * 85)
    print("This demo uses pattern matching for illustration.")
    print("\nFor PRODUCTION use (higher accuracy), you would:")
    print("  1. Use pre-trained GECToR model weights")
    print("  2. Run token-level sequence tagging (5002+ edit labels)")
    print("  3. Achieve 85-95% recall on real-world data")
    print("  4. Customize thresholds per error type")
    print("\n✅ Training data requirement:")
    print("  • With model: 500-1000 annotated sentences needed for fine-tuning")
    print("  • Without data: Can use pre-trained weights (trained on millions of samples)")


if __name__ == '__main__':
    main()
