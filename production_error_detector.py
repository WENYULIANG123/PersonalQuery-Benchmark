"""
Production-ready English Error Detection System
生产级英文错误检测系统
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    SPELLING = 'SPELLING'
    GRAMMAR = 'GRAMMAR'
    PUNCTUATION = 'PUNCTUATION'
    WORD_CHOICE = 'WORD_CHOICE'
    DELETION = 'DELETION'
    MERGE = 'MERGE'


@dataclass
class DetectedError:
    position: int
    token: str
    error_type: ErrorType
    confidence: float
    correction: Optional[str] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'token': self.token,
            'type': self.error_type.value,
            'confidence': round(self.confidence, 3),
            'correction': self.correction,
            'explanation': self.explanation
        }


@dataclass
class SentenceAnalysis:
    sentence: str
    tokens: List[str]
    errors: List[DetectedError]
    error_count: int
    quality_score: float
    
    def to_dict(self) -> Dict:
        return {
            'sentence': self.sentence,
            'num_errors': self.error_count,
            'quality_score': round(self.quality_score, 3),
            'errors': [e.to_dict() for e in self.errors]
        }


class RuleSet:
    def __init__(self):
        self.spelling_rules = self._build_spelling_rules()
        self.grammar_rules = self._build_grammar_rules()
        self.punctuation_rules = self._build_punctuation_rules()
    
    def _build_spelling_rules(self) -> Dict[str, str]:
        return {
            'teh': 'the',
            'recieved': 'received',
            'occured': 'occurred',
            'begining': 'beginning',
            'excelent': 'excellent',
            'seperate': 'separate',
            'occassion': 'occasion',
            'untill': 'until',
            'alot': 'a lot',
            'wiht': 'with',
            'writting': 'writing',
            'wich': 'which',
            'thier': 'their',
            'recieve': 'receive',
            'adress': 'address',
            'aquire': 'acquire',
            'accomodate': 'accommodate',
            'wich': 'which',
        }
    
    def _build_grammar_rules(self) -> Dict:
        return {
            'subject_verb_agreement': [
                {'pattern': ('i', 'likes'), 'correction': 'like', 'severity': 0.90},
                {'pattern': ('he', 'like'), 'correction': 'likes', 'severity': 0.90},
                {'pattern': ('she', 'like'), 'correction': 'likes', 'severity': 0.90},
                {'pattern': ('it', 'like'), 'correction': 'likes', 'severity': 0.90},
            ],
            'tense_consistency': [
                {'pattern': ('was', 'am'), 'issue': 'Tense mismatch', 'severity': 0.85},
            ]
        }
    
    def _build_punctuation_rules(self) -> Dict:
        return {
            'missing_comma_between_adjectives': [
                'quality good',
                'good price',
                'fast delivery',
            ]
        }


class EnglishErrorDetector:
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.rules = RuleSet()
        logger.info(f"Initialized ErrorDetector with confidence threshold: {min_confidence}")
    
    def detect(self, sentences: List[str]) -> List[SentenceAnalysis]:
        results = []
        for idx, sentence in enumerate(sentences):
            analysis = self._analyze_sentence(sentence, idx)
            results.append(analysis)
        return results
    
    def _analyze_sentence(self, sentence: str, idx: int = 0) -> SentenceAnalysis:
        tokens = sentence.split()
        errors = []
        
        errors.extend(self._detect_spelling_errors(tokens))
        errors.extend(self._detect_grammar_errors(tokens))
        errors.extend(self._detect_punctuation_errors(tokens, sentence))
        
        errors = [e for e in errors if e.confidence >= self.min_confidence]
        errors = self._deduplicate_errors(errors)
        errors = sorted(errors, key=lambda e: e.position)
        
        quality_score = self._calculate_quality_score(len(tokens), len(errors))
        
        return SentenceAnalysis(
            sentence=sentence,
            tokens=tokens,
            errors=errors,
            error_count=len(errors),
            quality_score=quality_score
        )
    
    def _detect_spelling_errors(self, tokens: List[str]) -> List[DetectedError]:
        errors = []
        for idx, token in enumerate(tokens):
            token_clean = token.lower().rstrip('.,!?;:\'"')
            
            if token_clean in self.rules.spelling_rules:
                correction = self.rules.spelling_rules[token_clean]
                errors.append(DetectedError(
                    position=idx,
                    token=token,
                    error_type=ErrorType.SPELLING,
                    confidence=0.95,
                    correction=correction,
                    explanation=f'Misspelled: "{token}" should be "{correction}"'
                ))
        return errors
    
    def _detect_grammar_errors(self, tokens: List[str]) -> List[DetectedError]:
        errors = []
        
        for idx in range(len(tokens) - 1):
            word1 = tokens[idx].lower()
            word2 = tokens[idx + 1].lower().rstrip('.,!?;:\'"')
            
            for rule in self.rules.grammar_rules['subject_verb_agreement']:
                if (word1, word2) == rule['pattern']:
                    errors.append(DetectedError(
                        position=idx + 1,
                        token=tokens[idx + 1],
                        error_type=ErrorType.GRAMMAR,
                        confidence=rule['severity'],
                        correction=rule['correction'],
                        explanation=f'Subject-verb agreement: should be "{rule["correction"]}"'
                    ))
        
        return errors
    
    def _detect_punctuation_errors(self, tokens: List[str], sentence: str) -> List[DetectedError]:
        errors = []
        sentence_lower = sentence.lower()
        
        for pattern in self.rules.punctuation_rules['missing_comma_between_adjectives']:
            if pattern in sentence_lower:
                pos = sentence_lower.find(pattern)
                token_idx = len(sentence[:pos].split()) - 1
                errors.append(DetectedError(
                    position=token_idx,
                    token=pattern,
                    error_type=ErrorType.PUNCTUATION,
                    confidence=0.85,
                    explanation='Missing comma between adjectives'
                ))
        
        return errors
    
    @staticmethod
    def _deduplicate_errors(errors: List[DetectedError]) -> List[DetectedError]:
        seen_positions = {}
        for error in errors:
            if error.position not in seen_positions:
                seen_positions[error.position] = error
            else:
                if error.confidence > seen_positions[error.position].confidence:
                    seen_positions[error.position] = error
        return list(seen_positions.values())
    
    @staticmethod
    def _calculate_quality_score(num_tokens: int, num_errors: int) -> float:
        if num_tokens == 0:
            return 0.0
        error_rate = num_errors / num_tokens
        quality_score = max(0.0, 1.0 - (error_rate * 0.2))
        return quality_score


class ErrorDetectionPipeline:
    def __init__(self, min_confidence: float = 0.5):
        self.detector = EnglishErrorDetector(min_confidence)
    
    def process(self, texts: List[str]) -> List[Dict]:
        analyses = self.detector.detect(texts)
        return [a.to_dict() for a in analyses]
    
    def process_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.process(batch)
            results.extend(batch_results)
        return results


def print_analysis_results(analyses: List[SentenceAnalysis]):
    print("\n" + "="*100)
    print("ENGLISH ERROR DETECTION - ANALYSIS RESULTS")
    print("="*100)
    
    total_errors = sum(a.error_count for a in analyses)
    avg_quality = sum(a.quality_score for a in analyses) / len(analyses) if analyses else 0
    
    for analysis in analyses:
        quality_indicator = "✓" if analysis.error_count == 0 else "⚠"
        print(f"\n{quality_indicator} {analysis.sentence}")
        print(f"   Quality Score: {analysis.quality_score:.1%} | Errors Found: {analysis.error_count}")
        
        if analysis.errors:
            print(f"\n   {'Pos':<4} {'Token':<18} {'Type':<15} {'Correction':<18} {'Conf.':<6}")
            print(f"   {'-'*94}")
            
            for error in analysis.errors:
                pos = str(error.position)
                token = error.token[:17]
                err_type = error.error_type.value
                correction = error.correction[:17] if error.correction else "-"
                conf = f"{error.confidence:.0%}"
                print(f"   {pos:<4} {token:<18} {err_type:<15} {correction:<18} {conf:<6}")
    
    print("\n" + "="*100)
    print(f"SUMMARY")
    print(f"  Total sentences: {len(analyses)}")
    print(f"  Total errors: {total_errors}")
    print(f"  Average quality: {avg_quality:.1%}")
    print("="*100 + "\n")


if __name__ == '__main__':
    test_sentences = [
        "This product is excellent with great quality.",
        "Teh product quality is really good.",
        "I likes this product very much.",
        "Great quality good price excellent service.",
        "I recieved the package very quickly.",
        "The delivery is very fast but product quality are bad.",
        "Fast shipping and good packaging makes this perfect.",
        "Product arrived excelent condition with fast delivery.",
    ]
    
    logger.info("="*80)
    logger.info("PRODUCTION ERROR DETECTION SYSTEM - TEST")
    logger.info("="*80)
    
    pipeline = ErrorDetectionPipeline(min_confidence=0.5)
    analyses = pipeline.detector.detect(test_sentences)
    
    print_analysis_results(analyses)
    
    json_results = pipeline.process(test_sentences)
    print("\nJSON Output Sample:")
    print(json.dumps(json_results[1:3], indent=2, ensure_ascii=False))
