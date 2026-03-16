#!/usr/bin/env python3
"""
Test script for three-way preference classifier

Tests the PreferenceClassifier on real user data to verify:
1. Explicit preferences are correctly identified (positive sentiment)
2. Implicit preferences are extracted from negative feedback
3. Conflicting preferences are detected within same dimension
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ..core.preference_classifier import PreferenceClassifierV2 as PreferenceClassifier, build_three_way_persona_context_v2 as build_three_way_persona_context


def test_basic_classification():
    """Test basic three-way classification on real data"""
    
    user_id = "A13OFOB1394G31"
    category = "Die-Cuts"
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    # Example selected_attributes from a real query
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"},
        {"dimension": "Product_Category", "value": "butterfly dies"},
        {"dimension": "Compatibility", "value": "metal shim"}
    ]
    
    print("=" * 80)
    print("TEST: Basic Three-way Classification")
    print("=" * 80)
    print(f"User: {user_id}")
    print(f"Category: {category}")
    print(f"Selected Dimensions: {[attr['dimension'] for attr in selected_attrs]}")
    print()
    
    # Initialize classifier
    classifier = PreferenceClassifier(user_id, processing_dir)
    
    # Classify preferences
    result = classifier.classify_query_preferences(category, selected_attrs)
    
    # Print results for each dimension
    for dimension, categories in result.items():
        print(f"\n{'='*60}")
        print(f"Dimension: {dimension}")
        print(f"{'='*60}")
        
        print(f"\n✅ EXPLICIT (Positive): {len(categories['explicit'])} items")
        for attr in categories['explicit'][:3]:  # Show first 3
            print(f"   - {attr.get('attribute', '')}")
            print(f"     Evidence: \"{attr.get('original_text', '')[:80]}...\"")
        
        print(f"\n⚠️  IMPLICIT (Negative → Improvement): {len(categories['implicit'])} items")
        for attr in categories['implicit'][:3]:
            print(f"   - Dislikes: {attr.get('attribute', '')}")
            improvement = attr.get('improvement_wish', '')
            if improvement:
                print(f"     → Expects: {improvement[:80]}")
        
        print(f"\n⚔️  CONFLICTING (Contradictory): {len(categories['conflicting'])} items")
        for attr in categories['conflicting'][:3]:
            print(f"   - {attr.get('attribute', '')} (sentiment: {attr.get('sentiment', '')})")
    
    print("\n" + "=" * 80)
    print("Formatted LLM Context:")
    print("=" * 80)
    
    formatted = classifier.format_classified_preferences(result, selected_attrs)
    print(formatted)
    
    return result


def test_conflict_detection():
    """Test conflict detection for same dimension with mixed sentiments"""
    
    print("\n\n" + "=" * 80)
    print("TEST: Conflict Detection")
    print("=" * 80)
    
    user_id = "A13OFOB1394G31"
    category = "Die-Cuts"
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    # Test with dimensions known to have conflicts
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Sizzix"},  # User has mixed feelings
        {"dimension": "Size_Dimensions", "value": "small"}     # User dislikes small
    ]
    
    classifier = PreferenceClassifier(user_id, processing_dir)
    result = classifier.classify_query_preferences(category, selected_attrs)
    
    # Check for conflicts
    has_conflicts = any(
        len(cats['conflicting']) > 0 
        for cats in result.values()
    )
    
    print(f"\nConflicts detected: {'YES ⚠️' if has_conflicts else 'NO ✅'}")
    
    for dimension, categories in result.items():
        if categories['conflicting']:
            print(f"\n❌ Dimension '{dimension}' has conflicts:")
            print(f"   Explicit: {len(categories['explicit'])} positive")
            print(f"   Implicit: {len(categories['implicit'])} negative")
            print(f"   Conflicting: {len(categories['conflicting'])} contradictory")
    
    return result


def test_full_context_generation():
    """Test full persona context generation for LLM"""
    
    print("\n\n" + "=" * 80)
    print("TEST: Full LLM Context Generation")
    print("=" * 80)
    
    user_id = "A13OFOB1394G31"
    category = "Die-Cuts"
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    # Simulate a query's selected_attributes
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"},
        {"dimension": "Functionality", "value": "mixing and matching"}
    ]
    
    # Build context
    context = build_three_way_persona_context(
        category, 
        selected_attrs, 
        user_id, 
        processing_dir
    )
    
    print("\nGenerated LLM Context:")
    print(context)
    
    return context


def compare_with_old_method():
    """Compare three-way classification with old persona_utils method"""
    
    print("\n\n" + "=" * 80)
    print("TEST: Comparison with Old Method")
    print("=" * 80)
    
    user_id = "A13OFOB1394G31"
    category = "Die-Cuts"
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"}
    ]
    
    # New method
    print("\n📊 NEW METHOD (Three-way Classification):")
    classifier = PreferenceClassifier(user_id, processing_dir)
    new_result = classifier.classify_query_preferences(category, selected_attrs)
    
    for dimension, cats in new_result.items():
        print(f"\n{dimension}:")
        print(f"  Explicit: {len(cats['explicit'])}")
        print(f"  Implicit: {len(cats['implicit'])}")
        print(f"  Conflicting: {len(cats['conflicting'])}")
    
    # Old method (from persona_utils.py)
    print("\n📊 OLD METHOD (Binary Classification):")
    try:
        from persona_utils import build_enhanced_persona_context
        
        query_info = {
            'query': "test query",
            'category': category,
            'selected_attributes': selected_attrs
        }
        
        old_context = build_enhanced_persona_context(
            category,
            selected_attrs,
            user_id,
            query_info,
            processing_dir
        )
        
        print(old_context[:500] + "...")
    except Exception as e:
        print(f"Could not load old method: {e}")
    
    print("\n✅ Key Improvement: New method explicitly separates:")
    print("   - Explicit (what user WANTS)")
    print("   - Implicit (what user wants to AVOID + improvements)")
    print("   - Conflicting (contradictory signals)")


def save_classification_examples():
    """Save classification examples to JSON for documentation"""
    
    print("\n\n" + "=" * 80)
    print("Saving Classification Examples...")
    print("=" * 80)
    
    user_id = "A13OFOB1394G31"
    category = "Die-Cuts"
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"},
        {"dimension": "Product_Category", "value": "butterfly dies"}
    ]
    
    classifier = PreferenceClassifier(user_id, processing_dir)
    result = classifier.classify_query_preferences(category, selected_attrs)
    
    # Save to file
    output_file = "/home/wlia0047/ar57/wenyu/result/personal_query/14_rerank/classification_example.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'user_id': user_id,
        'category': category,
        'selected_attributes': selected_attrs,
        'classification_result': result,
        'formatted_context': classifier.format_classified_preferences(result, selected_attrs)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {output_file}")


if __name__ == "__main__":
    print("🧪 Testing Three-way Preference Classifier")
    print("=" * 80)
    
    try:
        # Run all tests
        test_basic_classification()
        test_conflict_detection()
        test_full_context_generation()
        compare_with_old_method()
        save_classification_examples()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
