#!/usr/bin/env python3
"""
Improved Persona Context Builder with Conflict Resolution
Used by all LLM evaluation scripts
"""

import os
import re
import json
from typing import Dict, List, Optional


def classify_preference_relevance(preference: Dict, query_info: Dict) -> str:
    """
    Classify preference relevance: REQUIRED, RELEVANT, CONFLICTING, IRRELEVANT
    
    Args:
        preference: Dict with keys 'dimension', 'attribute'/'value', 'sentiment'
        query_info: Dict with keys 'query', 'category', 'selected_attributes'
    
    Returns:
        One of: 'REQUIRED', 'RELEVANT', 'CONFLICTING', 'IRRELEVANT'
    """
    dim = preference.get('dimension', '')
    value = preference.get('attribute', preference.get('value', '')).lower()
    sentiment = preference.get('sentiment', 'neutral')
    
    query_text = query_info.get('query', '').lower()
    query_category = query_info.get('category', '').lower()
    
    # Check if directly mentioned in query
    if value in query_text:
        if sentiment == 'negative':
            return 'CONFLICTING'
        return 'REQUIRED'
    
    # Check specific conflicts
    if dim == 'Brand_Preference':
        # Check for Sizzix/Big Shot conflict
        if 'sizzix' in value and 'big shot' in query_text and sentiment == 'negative':
            return 'CONFLICTING'
        # Check if query asks for a different brand
        for attr in query_info.get('selected_attributes', []):
            if attr.get('dimension') == 'Brand_Preference' and attr.get('value', '').lower() != value:
                return 'IRRELEVANT'
    
    # Check if same product category
    if dim == 'Product_Category':
        if 'die' in value and 'die' in query_category:
            return 'RELEVANT'
        elif value not in query_text and value not in query_category:
            return 'IRRELEVANT'
    
    # Check functionality relevance
    if dim == 'Functionality':
        # Specific functionalities that don't match query
        irrelevant_functions = ['flower pot', 'reindeer', 'butterfly']
        if any(func in value for func in irrelevant_functions) and not any(func in query_text for func in irrelevant_functions):
            return 'IRRELEVANT'
    
    # Check packaging quantity
    if dim == 'Packaging_Quantity':
        # Check if quantities match
        query_numbers = re.findall(r'\d+', query_text)
        pref_numbers = re.findall(r'\d+', value)
        if query_numbers and pref_numbers and query_numbers[0] != pref_numbers[0]:
            return 'IRRELEVANT'
    
    # Check compatibility
    if dim == 'Compatibility':
        # Check for specific conflicts
        if 'metal shim' in value and 'shim' in query_text:
            return 'RELEVANT'
        if sentiment == 'negative' and any(neg in value for neg in ['movers & shapers', 'cutting pads']):
            # If user explicitly doesn't want something that's not in query
            if not any(neg in query_text for neg in ['movers', 'shapers', 'cutting pad']):
                return 'IRRELEVANT'
    
    # Default to relevant if in selected dimensions
    selected_dims = {attr.get('dimension', '') for attr in query_info.get('selected_attributes', [])}
    if dim in selected_dims:
        return 'RELEVANT'
    
    return 'IRRELEVANT'


def load_processing_attrs(category: str, user_id: str, processing_dir: str) -> List:
    """Load processing attributes for persona context"""
    if not processing_dir:
        return []
    category_filename = category.replace(" & ", "_and_").replace(" ", "_")
    processing_file = os.path.join(processing_dir, f"persona_{category_filename}_{user_id}.json")
    if not os.path.exists(processing_file):
        return []
    try:
        with open(processing_file, 'r') as f:
            data = json.load(f)
            return data.get('attributes', [])
    except Exception as e:
        print(f"Warning: Failed to load processing attrs: {e}")
        return []


def build_enhanced_persona_context(category: str, selected_attributes: List, user_id: str, query_info: Dict, processing_dir: str) -> str:
    """
    Build persona context with relevance classification and conflict handling
    
    Args:
        category: Product category
        selected_attributes: List of selected attributes from query
        user_id: User ID
        query_info: Dict with query details
        processing_dir: Directory containing persona files
    
    Returns:
        Formatted persona context string
    """
    import json
    
    if not selected_attributes:
        return ""
    
    selected_dims = set(attr.get('dimension', '') for attr in selected_attributes if attr.get('dimension'))
    all_attrs = load_processing_attrs(category, user_id, processing_dir)
    
    if not all_attrs:
        # Fallback to simple format
        contexts = [f"  - {attr.get('dimension', '')}: {attr.get('value', '')}" 
                    for attr in selected_attributes if attr.get('dimension')]
        return "User Preferences:\n" + "\n".join(contexts) if contexts else ""
    
    # Classify all attributes
    classified = {
        'REQUIRED': [],
        'RELEVANT': [],
        'CONFLICTING': [],
        'IRRELEVANT': []
    }
    
    attrs_by_dim = {}
    for attr in all_attrs:
        dim = attr.get('dimension', '')
        if dim in selected_dims:
            if dim not in attrs_by_dim:
                attrs_by_dim[dim] = []
            attrs_by_dim[dim].append(attr)
    
    # Classify each preference
    for dim in attrs_by_dim:
        for attr in attrs_by_dim[dim]:
            relevance = classify_preference_relevance(attr, query_info)
            classified[relevance].append(attr)
    
    # Build context with classified preferences
    contexts = []
    
    if classified['REQUIRED']:
        contexts.append("Query-Required Preferences:")
        for attr in classified['REQUIRED']:
            sentiment = attr.get('sentiment', 'neutral')
            contexts.append(f"  - {attr['dimension']}: {attr.get('attribute', '')} (sentiment: {sentiment})")
    
    if classified['RELEVANT']:
        contexts.append("\nRelevant Preferences:")
        for attr in classified['RELEVANT']:
            sentiment = attr.get('sentiment', 'neutral')
            contexts.append(f"  - {attr['dimension']}: {attr.get('attribute', '')} (sentiment: {sentiment})")
    
    if classified['CONFLICTING']:
        contexts.append("\nConflicting Preferences (IGNORE - Query requirements take priority):")
        for attr in classified['CONFLICTING']:
            sentiment = attr.get('sentiment', 'neutral')
            contexts.append(f"  - {attr['dimension']}: {attr.get('attribute', '')} (sentiment: {sentiment}) [CONFLICTS WITH QUERY]")
    
    # Don't include irrelevant preferences
    
    return "\n".join(contexts) if contexts else ""


def build_improved_prompt(query_info: Dict, doc_text: str, persona_context: str) -> str:
    """
    Build improved prompt with positive reinforcement and better scoring rules
    
    Args:
        query_info: Dict with query details
        doc_text: Product document text
        persona_context: Formatted persona context (can be empty for standard mode)
    
    Returns:
        Complete prompt string
    """
    if persona_context:
        prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[User Profile]
{persona_context}

[Key Guidelines]
1. If a product matches the main intent (brand + category + compatibility), it is RELEVANT even if some details are missing.
2. For die-cut products, piece counts and specific item lists are often NOT in titles - absence is not evidence of mismatch.
3. "Baby clothes" in query matches "Baby Boy Clothes" in product - partial match is still a match.
4. Ignore conflicting preferences - they do not reduce relevance.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query_info['query']}"
Product Info:
{doc_text}

Please analyze what RELEVANT features this product has:
1. List all matching elements (brand, category, compatibility, item types)
2. For each query requirement, note if it's met, partially met, or not met
3. Give credit for partial matches (e.g., "baby clothes" covers part of "baby clothes, toy bunny, clothes pins")
4. End with "Final Score: X.X" where X.X reflects overall relevance

Analysis:'''
    else:
        prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query_info['query']}"
Product Info:
{doc_text}

Please analyze what RELEVANT features this product has:
1. List all matching elements (brand, category, compatibility, item types)
2. For each query requirement, note if it's met, partially met, or not met
3. Give credit for partial matches
4. End with "Final Score: X.X" where X.X reflects overall relevance

Analysis:'''
    
    return prompt


# Alias for backward compatibility
build_persona_context = build_enhanced_persona_context
