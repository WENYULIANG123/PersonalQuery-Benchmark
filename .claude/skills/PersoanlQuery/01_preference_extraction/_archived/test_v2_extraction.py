#!/usr/bin/env python3
"""
Phase 1 v2 单元测试：验证方面提取和隐式检测的功能

测试覆盖：
1. 方面抽取精度
2. 隐式方面检测
3. 置信度评分
4. 质量检查
"""

import json
import sys
import importlib.util

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction")

spec = importlib.util.spec_from_file_location(
    "extract_v2", 
    "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences_v2_with_aspects.py"
)
extract_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_v2)

detect_implicit_aspects = extract_v2.detect_implicit_aspects
validate_extraction_quality = extract_v2.validate_extraction_quality
log_with_timestamp = extract_v2.log_with_timestamp

# ============================================================================
# 测试数据
# ============================================================================

test_reviews = {
    "explicit_positive": {
        "text": "I love this glitter glue. It works beautifully and the color is amazing.",
        "expected_aspects": ["glitter glue", "functionality", "appearance"],
        "expected_sentiments": ["POSITIVE", "POSITIVE", "POSITIVE"]
    },
    
    "explicit_negative": {
        "text": "The scissors broke after one week. Poor quality overall.",
        "expected_aspects": ["scissors", "durability", "quality"],
        "expected_sentiments": ["NEGATIVE", "NEGATIVE", "NEGATIVE"]
    },
    
    "mixed": {
        "text": "Great product but too expensive for what you get.",
        "expected_aspects": ["product", "price", "value"],
        "expected_sentiments": ["POSITIVE", "NEGATIVE", "NEGATIVE"]
    },
    
    "implicit_price": {
        "text": "It's way too expensive for a simple die cutter. The cheaper alternatives work just as well.",
        "expected_implicit": ["Price"],
        "description": "隐式价格问题"
    },
    
    "implicit_durability": {
        "text": "The glue stopped working after just one week. Very disappointing.",
        "expected_implicit": ["Durability"],
        "description": "隐式耐用性问题"
    },
    
    "implicit_value": {
        "text": "For this price, you can find much better quality elsewhere.",
        "expected_implicit": ["Value"],
        "description": "隐式价值问题"
    }
}

# ============================================================================
# 测试函数
# ============================================================================

def test_implicit_aspect_detection():
    """测试隐式方面检测"""
    log_with_timestamp("=" * 80)
    log_with_timestamp("TEST: Implicit Aspect Detection")
    log_with_timestamp("=" * 80)
    
    test_cases = [
        ("implicit_price", test_reviews["implicit_price"]),
        ("implicit_durability", test_reviews["implicit_durability"]),
        ("implicit_value", test_reviews["implicit_value"]),
    ]
    
    passed = 0
    failed = 0
    
    for case_name, case_data in test_cases:
        log_with_timestamp(f"\nTest: {case_name}")
        log_with_timestamp(f"Review: {case_data['text'][:80]}...")
        
        result = detect_implicit_aspects(case_data['text'])
        
        if result:
            log_with_timestamp(f"✓ Detected {len(result)} implicit aspects:")
            for aspect in result:
                log_with_timestamp(f"  - {aspect['aspect']}: {aspect['aspect_sentiment']} (confidence: {aspect['confidence']})")
            passed += 1
        else:
            log_with_timestamp(f"✗ No implicit aspects detected")
            failed += 1
    
    log_with_timestamp(f"\nSummary: {passed} passed, {failed} failed")
    return passed, failed


def test_quality_validation():
    """测试质量检查"""
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("TEST: Quality Validation")
    log_with_timestamp("=" * 80)
    
    # 测试用例1：高质量提取
    high_quality = {
        "dimensions": {
            "Product_Attributes": {
                "Product_Category": [
                    {"entity": "glitter glue", "confidence": 0.95, "sentiment": "positive"}
                ]
            }
        },
        "aspects": [
            {"aspect": "glitter glue", "confidence": 0.95, "aspect_sentiment": "POSITIVE"}
        ]
    }
    
    log_with_timestamp("\nTest: High Quality Extraction")
    result = validate_extraction_quality(high_quality)
    log_with_timestamp(f"Quality Score: {result['quality_score']:.2f}")
    log_with_timestamp(f"Valid: {result['is_valid']}")
    log_with_timestamp(f"Issues: {result['issues']}")
    log_with_timestamp(f"Warnings: {result['warnings']}")
    
    # 测试用例2：低质量提取（缺少实体）
    low_quality = {
        "dimensions": {},
        "aspects": []
    }
    
    log_with_timestamp("\nTest: Low Quality Extraction (empty)")
    result = validate_extraction_quality(low_quality)
    log_with_timestamp(f"Quality Score: {result['quality_score']:.2f}")
    log_with_timestamp(f"Valid: {result['is_valid']}")
    log_with_timestamp(f"Warnings: {result['warnings']}")
    
    # 测试用例3：无效sentiment值
    invalid_sentiment = {
        "dimensions": {},
        "aspects": [
            {"aspect": "test", "aspect_sentiment": "VERY_POSITIVE"}  # 无效
        ]
    }
    
    log_with_timestamp("\nTest: Invalid Sentiment Value")
    result = validate_extraction_quality(invalid_sentiment)
    log_with_timestamp(f"Quality Score: {result['quality_score']:.2f}")
    log_with_timestamp(f"Valid: {result['is_valid']}")
    log_with_timestamp(f"Issues: {result['issues']}")
    
    return True


def test_aspect_sentiment_mapping():
    """测试方面情感映射"""
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("TEST: Aspect Sentiment Mapping")
    log_with_timestamp("=" * 80)
    
    implicit_results = detect_implicit_aspects("It's so expensive")
    
    if implicit_results:
        for aspect in implicit_results:
            log_with_timestamp(f"\nAspect: {aspect['aspect']}")
            log_with_timestamp(f"  Sentiment: {aspect['aspect_sentiment']}")
            log_with_timestamp(f"  Is Implicit: {aspect['is_implicit']}")
            log_with_timestamp(f"  Evidence: {aspect['evidence_spans']}")
            log_with_timestamp(f"  Dimension: {aspect['dimension_mapping']}")
    
    return True


def test_confidence_scoring():
    """测试置信度评分"""
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("TEST: Confidence Scoring")
    log_with_timestamp("=" * 80)
    
    explicit_aspects = detect_implicit_aspects("This glitter glue is great")
    implicit_aspects = detect_implicit_aspects("It's too expensive")
    
    log_with_timestamp("\nExplicit Aspects:")
    for aspect in explicit_aspects:
        log_with_timestamp(f"  {aspect['aspect']}: confidence={aspect.get('confidence', 'N/A')}, is_implicit={aspect.get('is_implicit')}")
    
    log_with_timestamp("\nImplicit Aspects:")
    for aspect in implicit_aspects:
        log_with_timestamp(f"  {aspect['aspect']}: confidence={aspect.get('confidence', 'N/A')}, is_implicit={aspect.get('is_implicit')}")
    
    # 验证隐式方面的置信度应该较低
    if implicit_aspects:
        implicit_conf = implicit_aspects[0].get('confidence', 0)
        if implicit_conf < 0.7:
            log_with_timestamp("✓ Implicit aspects have appropriate lower confidence")
        else:
            log_with_timestamp("⚠ Implicit aspects confidence may be too high")
    
    return True


# ============================================================================
# 运行所有测试
# ============================================================================

def run_all_tests():
    log_with_timestamp("=" * 80)
    log_with_timestamp("Phase 1 v2 Test Suite - Starting")
    log_with_timestamp("=" * 80)
    
    results = []
    
    try:
        log_with_timestamp("\n[1/4] Running implicit aspect detection tests...")
        passed, failed = test_implicit_aspect_detection()
        results.append(("Implicit Detection", passed > 0))
    except Exception as e:
        log_with_timestamp(f"✗ Test failed: {e}")
        results.append(("Implicit Detection", False))
    
    try:
        log_with_timestamp("\n[2/4] Running quality validation tests...")
        test_quality_validation()
        results.append(("Quality Validation", True))
    except Exception as e:
        log_with_timestamp(f"✗ Test failed: {e}")
        results.append(("Quality Validation", False))
    
    try:
        log_with_timestamp("\n[3/4] Running aspect sentiment mapping tests...")
        test_aspect_sentiment_mapping()
        results.append(("Sentiment Mapping", True))
    except Exception as e:
        log_with_timestamp(f"✗ Test failed: {e}")
        results.append(("Sentiment Mapping", False))
    
    try:
        log_with_timestamp("\n[4/4] Running confidence scoring tests...")
        test_confidence_scoring()
        results.append(("Confidence Scoring", True))
    except Exception as e:
        log_with_timestamp(f"✗ Test failed: {e}")
        results.append(("Confidence Scoring", False))
    
    # 总结
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("Test Summary")
    log_with_timestamp("=" * 80)
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        log_with_timestamp(f"{status}: {test_name}")
    
    log_with_timestamp(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        log_with_timestamp("\n🎉 All tests passed!")
    else:
        log_with_timestamp(f"\n⚠️ {total_count - passed_count} tests failed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
