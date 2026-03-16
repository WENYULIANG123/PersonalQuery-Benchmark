#!/usr/bin/env python3
"""
Example: Using Three-way Preference Classifier in LLM Reranking

This script demonstrates how to integrate the three-way preference classifier
into an LLM reranking workflow (GLM, Minimax, Qwen, etc.)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ..core.preference_classifier import build_three_way_persona_context_v2 as build_three_way_persona_context


def example_1_basic_usage():
    """
    Example 1: Basic usage - Generate persona context for a single query
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Query information (from Stage 7 or Stage 10)
    query_data = {
        'query': 'I need Spellbinders balloon dies with clean cutting performance',
        'category': 'Die-Cuts',
        'selected_attributes': [
            {'dimension': 'Brand_Preference', 'value': 'Spellbinders'},
            {'dimension': 'Product_Category', 'value': 'balloon dies'},
            {'dimension': 'Performance', 'value': 'clean cutting'}
        ]
    }
    
    # Generate three-way classified persona context
    context = build_three_way_persona_context(
        category=query_data['category'],
        selected_attributes=query_data['selected_attributes'],
        user_id='A13OFOB1394G31',
        processing_dir='/home/wlia0047/ar57/wenyu/result/personal_query/03_processing'
    )
    
    print(f"\nQuery: {query_data['query']}")
    print(f"\nGenerated Persona Context:\n")
    print(context)
    
    return context


def example_2_build_llm_prompt():
    """
    Example 2: Build complete LLM reranking prompt with persona context
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Complete LLM Reranking Prompt")
    print("=" * 80)
    
    # Simulate query and product data
    query = "Spellbinders butterfly dies for my Grand Calibur machine"
    product_text = """
    Title: Spellbinders S4-310 Shapeabilities Botanical Swirls and Accents Die Template
    Brand: Spellbinders
    Features:
    - Perfectly suited for your Spellbinders Grand Calibur machine
    - Creates intricate butterfly and floral designs
    - Clean, precise cutting edge
    - Compatible with most die-cutting machines
    Price: $19.99
    """
    
    # Get persona context
    persona_context = build_three_way_persona_context(
        category='Die-Cuts',
        selected_attributes=[
            {'dimension': 'Brand_Preference', 'value': 'Spellbinders'},
            {'dimension': 'Product_Category', 'value': 'butterfly dies'},
            {'dimension': 'Compatibility', 'value': 'Grand Calibur'}
        ],
        user_id='A13OFOB1394G31',
        processing_dir='/home/wlia0047/ar57/wenyu/result/personal_query/03_processing'
    )
    
    # Build complete prompt
    prompt = f"""You are an expert e-commerce search relevance evaluator.
Score how well this product matches the user's query and preferences on a scale of 0.0 to 1.0.

{persona_context}

[Scoring Guidelines]
- Give higher scores (0.8-1.0) if the product matches explicit preferences
- Consider implicit preferences (what user wants to avoid)
- Ignore conflicting preferences if query is clear
- Focus on alignment with user's stated needs

[Query]
{query}

[Product]
{product_text}

Analyze the match and provide:
1. Alignment with explicit preferences (what user WANTS)
2. Avoidance of implicit preferences (what user wants to AVOID)
3. Overall relevance score

Final Score: X.X
"""
    
    print(prompt)
    
    return prompt


def example_3_batch_reranking():
    """
    Example 3: Batch reranking for multiple products
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Batch Reranking Workflow")
    print("=" * 80)
    
    # Load query file (simulated)
    query_data = {
        'asin': 'B00HHEX8SS',
        'category': 'Die-Cuts',
        'query': 'balloon dies for Spellbinders machine',
        'selected_attributes': [
            {'dimension': 'Brand_Preference', 'value': 'Spellbinders'},
            {'dimension': 'Product_Category', 'value': 'balloon dies'}
        ]
    }
    
    # Simulated BM25 candidates (top-50)
    candidates = [
        {'asin': 'B001', 'title': 'Spellbinders Balloon Dies Set', 'score': 15.2},
        {'asin': 'B002', 'title': 'Sizzix Balloon Cutting Dies', 'score': 14.8},
        {'asin': 'B003', 'title': 'Spellbinders Animal Dies', 'score': 13.5},
        # ... more candidates
    ]
    
    # Generate persona context ONCE for all candidates
    persona_context = build_three_way_persona_context(
        category=query_data['category'],
        selected_attributes=query_data['selected_attributes'],
        user_id='A13OFOB1394G31',
        processing_dir='/home/wlia0047/ar57/wenyu/result/personal_query/03_processing'
    )
    
    print(f"\nQuery: {query_data['query']}")
    print(f"Number of candidates: {len(candidates)}")
    print(f"\nPersona Context (shared across all candidates):")
    print(persona_context[:300] + "...")
    
    # Rerank each candidate (simulated)
    print("\n\nReranking Results:")
    print("-" * 60)
    
    for i, candidate in enumerate(candidates[:3], 1):
        # In real code, you would call LLM API here with:
        # prompt = build_prompt(query, candidate['title'], persona_context)
        # score = llm_api.score(prompt)
        
        simulated_score = 0.95 if 'Spellbinders' in candidate['title'] else 0.65
        
        print(f"{i}. {candidate['title']}")
        print(f"   BM25 Score: {candidate['score']:.2f}")
        print(f"   LLM Score: {simulated_score:.2f}")
        print(f"   {'✓ Matches explicit preference (Spellbinders)' if 'Spellbinders' in candidate['title'] else '⚠ Does not match brand preference'}")
        print()
    
    return candidates


def example_4_comparison():
    """
    Example 4: Compare standard vs personalized reranking
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Standard vs Personalized Comparison")
    print("=" * 80)
    
    query = "balloon dies"
    
    # Standard reranking (no persona)
    standard_prompt = f"""
    Query: {query}
    Product: Spellbinders Balloon Dies
    
    Score relevance: 0.0 to 1.0
    """
    
    print("📊 STANDARD RERANKING (No Persona):")
    print(standard_prompt)
    print("→ Generic scoring, no personalization")
    
    # Personalized reranking (with three-way classification)
    persona_context = build_three_way_persona_context(
        category='Die-Cuts',
        selected_attributes=[
            {'dimension': 'Product_Category', 'value': 'balloon dies'}
        ],
        user_id='A13OFOB1394G31',
        processing_dir='/home/wlia0047/ar57/wenyu/result/personal_query/03_processing'
    )
    
    personalized_prompt = f"""
    {persona_context}
    
    Query: {query}
    Product: Spellbinders Balloon Dies
    
    Score relevance considering user preferences: 0.0 to 1.0
    """
    
    print("\n📊 PERSONALIZED RERANKING (Three-way Classification):")
    print(personalized_prompt[:400] + "...")
    print("→ Knows user prefers Spellbinders, avoids Sizzix, expects clean cutting")
    
    print("\n✅ Key Advantage:")
    print("   Personalized version boosts Spellbinders products (explicit preference)")
    print("   and penalizes Sizzix products (implicit preference to avoid)")


def example_5_real_integration():
    """
    Example 5: Real integration pattern for existing rerankers
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 5: Integration Pattern for Existing Rerankers")
    print("=" * 80)
    
    code_example = '''
# In your existing reranker script (e.g., 13_evaluate_glm_5_both.py)

# OLD CODE (binary classification):
from persona_utils import build_enhanced_persona_context

persona_context = build_enhanced_persona_context(
    category, selected_attrs, user_id, query_info, processing_dir
)

# NEW CODE (three-way classification):
from ..core.preference_classifier import build_three_way_persona_context

persona_context = build_three_way_persona_context(
    category, selected_attrs, user_id, processing_dir
)

# Rest of your code stays the same!
# The persona_context is now structured with explicit/implicit/conflicting
'''
    
    print(code_example)
    
    print("\n✅ Migration is simple:")
    print("   1. Import build_three_way_persona_context instead of build_enhanced_persona_context")
    print("   2. Remove query_info parameter (not needed)")
    print("   3. Keep everything else the same")
    print("\n✅ Benefits:")
    print("   - Better structured context for LLM")
    print("   - Clear separation of positive/negative preferences")
    print("   - Automatic conflict detection")


if __name__ == "__main__":
    print("\n" + "🎯 " * 20)
    print("Three-way Preference Classifier - Usage Examples")
    print("🎯 " * 20 + "\n")
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_build_llm_prompt()
        example_3_batch_reranking()
        example_4_comparison()
        example_5_real_integration()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nFor more details, see:")
        print("  - README_PREFERENCE_CLASSIFIER.md")
        print("  - test_preference_classifier.py")
        print("  - preference_classifier.py (source code)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
