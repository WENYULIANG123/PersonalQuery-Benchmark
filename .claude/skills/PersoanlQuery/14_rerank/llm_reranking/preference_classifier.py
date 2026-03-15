#!/usr/bin/env python3
"""
Three-way Preference Classifier v2 (CORRECTED)

Classifies historical user preferences by comparing query attribute VALUES:
1. Explicit: Query attribute value has POSITIVE sentiment in history
2. Implicit: Query attribute value has NEGATIVE sentiment in history (conflict!)
3. Conflicting: Query attribute value has BOTH positive AND negative in history

Example:
    Query: {"dimension": "Brand", "value": "Sizzix"}
    History: [
        {"attribute": "Spellbinders", "sentiment": "positive"},  # Ignore (not "Sizzix")
        {"attribute": "Sizzix", "sentiment": "negative"},        # Match! → Implicit
    ]
    Result: Implicit (user historically dislikes Sizzix, but query asks for it)
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import re


class PreferenceClassifierV2:
    """
    Classifies user preferences by matching query attribute VALUES against history
    """
    
    def __init__(self, user_id: str, processing_dir: str):
        """
        Args:
            user_id: User ID
            processing_dir: Directory containing persona_*.json files from Stage 3
        """
        self.user_id = user_id
        self.processing_dir = processing_dir
        self.persona_cache = {}
    
    def load_persona_data(self, category: str) -> List[Dict]:
        """Load persona attributes for a specific category"""
        if category in self.persona_cache:
            return self.persona_cache[category]
        
        category_filename = category.replace(" & ", "_and_").replace(" ", "_")
        persona_file = os.path.join(
            self.processing_dir, 
            f"persona_{category_filename}_{self.user_id}.json"
        )
        
        if not os.path.exists(persona_file):
            print(f"⚠️ Persona file not found: {persona_file}")
            return []
        
        try:
            with open(persona_file, 'r') as f:
                data = json.load(f)
                attributes = data.get('attributes', [])
                self.persona_cache[category] = attributes
                return attributes
        except Exception as e:
            print(f"❌ Error loading persona file {persona_file}: {e}")
            return []
    
    def fuzzy_match_value(self, query_value: str, historical_value: str) -> bool:
        """
        Check if query value matches historical value (fuzzy matching)
        
        Args:
            query_value: Value from selected_attributes (e.g., "Sizzix")
            historical_value: Value from historical preference (e.g., "Sizzix Big Shot")
        
        Returns:
            True if they match (contains, fuzzy, or exact)
        """
        query_lower = query_value.lower().strip()
        hist_lower = historical_value.lower().strip()
        
        # Exact match
        if query_lower == hist_lower:
            return True
        
        # Contains (either direction)
        if query_lower in hist_lower or hist_lower in query_lower:
            return True
        
        # Token overlap (for multi-word values)
        query_tokens = set(re.findall(r'\w+', query_lower))
        hist_tokens = set(re.findall(r'\w+', hist_lower))
        
        # Require significant overlap (>= 50% of tokens)
        if query_tokens and hist_tokens:
            overlap = len(query_tokens & hist_tokens)
            min_tokens = min(len(query_tokens), len(hist_tokens))
            if overlap >= max(1, min_tokens * 0.5):
                return True
        
        return False
    
    def classify_single_attribute(
        self,
        query_attr: Dict,  # {"dimension": "Brand_Preference", "value": "Sizzix"}
        all_historical_attrs: List[Dict]  # All persona attributes
    ) -> Dict[str, List[Dict]]:
        """
        Classify a single query attribute against historical preferences
        
        Args:
            query_attr: Single attribute from selected_attributes
            all_historical_attrs: All user attributes from persona data
        
        Returns:
            {
                'explicit': [matched positive attributes],
                'implicit': [matched negative attributes],
                'conflicting': [matched attributes with both sentiments]
            }
        """
        dimension = query_attr.get('dimension')
        query_value = query_attr.get('value', '')
        
        if not dimension or not query_value:
            return {'explicit': [], 'implicit': [], 'conflicting': []}
        
        # Find all historical attributes in the same dimension
        dimension_attrs = [
            attr for attr in all_historical_attrs
            if attr.get('dimension') == dimension
        ]
        
        # Find attributes that match the query value
        matched_positive = []
        matched_negative = []
        
        for attr in dimension_attrs:
            hist_value = attr.get('attribute', attr.get('value', ''))
            sentiment = attr.get('sentiment', 'neutral')
            
            # Check if this historical attribute matches the query value
            if self.fuzzy_match_value(query_value, hist_value):
                if sentiment == 'positive':
                    matched_positive.append(attr)
                elif sentiment == 'negative':
                    matched_negative.append(attr)
        
        # Classify based on matches
        explicit = []
        implicit = []
        conflicting = []
        
        # If query value has BOTH positive and negative in history → Conflicting
        if matched_positive and matched_negative:
            conflicting = matched_positive + matched_negative
        # If query value only has positive in history → Explicit
        elif matched_positive:
            explicit = matched_positive
        # If query value only has negative in history → Implicit (conflict!)
        elif matched_negative:
            implicit = matched_negative
        
        return {
            'explicit': explicit,
            'implicit': implicit,
            'conflicting': conflicting
        }
    
    def classify_query_preferences(
        self,
        category: str,
        selected_attributes: List[Dict]
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Classify preferences for all attributes in selected_attributes
        
        Args:
            category: Product category
            selected_attributes: List of {"dimension": ..., "value": ...}
        
        Returns:
            {
                'Brand_Preference': {
                    'explicit': [...],
                    'implicit': [...],
                    'conflicting': [...]
                },
                ...
            }
        """
        # Load historical preferences
        all_attrs = self.load_persona_data(category)
        
        if not all_attrs:
            print(f"⚠️ No persona data found for category: {category}")
            return {}
        
        result = {}
        
        # Process each query attribute
        for query_attr in selected_attributes:
            dimension = query_attr.get('dimension')
            if not dimension:
                continue
            
            # Classify this specific attribute value
            classification = self.classify_single_attribute(query_attr, all_attrs)
            
            # Merge results for the same dimension
            if dimension not in result:
                result[dimension] = {
                    'explicit': [],
                    'implicit': [],
                    'conflicting': []
                }
            
            # Append (avoid duplicates)
            for category_key in ['explicit', 'implicit', 'conflicting']:
                for attr in classification[category_key]:
                    if attr not in result[dimension][category_key]:
                        result[dimension][category_key].append(attr)
        
        return result
    
    def format_classified_preferences(
        self,
        classified: Dict[str, Dict[str, List[Dict]]],
        selected_attributes: List[Dict]
    ) -> str:
        """
        Format classified preferences with query context
        
        Args:
            classified: Output from classify_query_preferences
            selected_attributes: Original query attributes for context
        
        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        
        for dimension, categories in classified.items():
            # Find query values for this dimension
            query_values = [
                attr.get('value', '') 
                for attr in selected_attributes 
                if attr.get('dimension') == dimension
            ]
            
            lines.append(f"\n### {dimension}")
            lines.append(f"Query asks for: {', '.join(query_values)}")
            
            # Explicit preferences (query value is positive in history)
            if categories['explicit']:
                lines.append("\n**✅ Explicit Preferences (Positive Match):**")
                lines.append("  → Query value has POSITIVE sentiment in user history")
                for attr in categories['explicit']:
                    value = attr.get('attribute', attr.get('value', ''))
                    evidence = attr.get('original_text', '')[:100]
                    lines.append(f"  ✓ {value}")
                    if evidence:
                        lines.append(f"    Evidence: \"{evidence}...\"")
            
            # Implicit preferences (query value is negative in history - CONFLICT!)
            if categories['implicit']:
                lines.append("\n**⚠️  Implicit Preferences (Negative Match - CONFLICT):**")
                lines.append("  → Query asks for something user DISLIKES in history!")
                for attr in categories['implicit']:
                    value = attr.get('attribute', attr.get('value', ''))
                    improvement = attr.get('improvement_wish', '')
                    evidence = attr.get('original_text', '')[:100]
                    lines.append(f"  ✗ {value} (user dislikes this)")
                    if evidence:
                        lines.append(f"    Evidence: \"{evidence}...\"")
                    if improvement:
                        lines.append(f"    → User expects: {improvement}")
            
            # Conflicting preferences (query value has BOTH positive and negative)
            if categories['conflicting']:
                lines.append("\n**⚔️  Conflicting Preferences (Mixed Signals):**")
                lines.append("  → Query value has BOTH positive AND negative in history")
                positive = [a for a in categories['conflicting'] if a.get('sentiment') == 'positive']
                negative = [a for a in categories['conflicting'] if a.get('sentiment') == 'negative']
                
                if positive:
                    lines.append("  Positive mentions:")
                    for attr in positive:
                        evidence = attr.get('original_text', '')[:80]
                        lines.append(f"    ✓ \"{evidence}...\"")
                
                if negative:
                    lines.append("  Negative mentions:")
                    for attr in negative:
                        evidence = attr.get('original_text', '')[:80]
                        lines.append(f"    ✗ \"{evidence}...\"")
                
                lines.append("  ⚠️ **Resolution**: Prioritize query intent if explicit")
        
        return "\n".join(lines) if lines else ""


def build_three_way_persona_context_v2(
    category: str,
    selected_attributes: List[Dict],
    user_id: str,
    processing_dir: str
) -> str:
    """
    Build persona context with corrected three-way classification
    
    Compares query attribute VALUES against historical evaluations of those VALUES
    
    Args:
        category: Product category
        selected_attributes: List of {"dimension": ..., "value": ...}
        user_id: User ID
        processing_dir: Directory containing persona files
    
    Returns:
        Formatted persona context string
    """
    if not selected_attributes:
        return ""
    
    classifier = PreferenceClassifierV2(user_id, processing_dir)
    classified = classifier.classify_query_preferences(category, selected_attributes)
    
    if not classified:
        return ""
    
    header = "User Preference Profile (Query-Value Matched Classification):\n"
    header += "=" * 70 + "\n"
    header += "NOTE: Classification based on historical sentiment for QUERY VALUES\n"
    header += "=" * 70
    
    body = classifier.format_classified_preferences(classified, selected_attributes)
    
    return header + body if body else ""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python preference_classifier_v2.py <user_id> <category>")
        print("Example: python preference_classifier_v2.py A13OFOB1394G31 Die-Cuts")
        sys.exit(1)
    
    user_id = sys.argv[1]
    category = sys.argv[2]
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    # Test case 1: Query asks for something user likes
    print("\n" + "=" * 70)
    print("TEST 1: Query asks for Spellbinders (user likes)")
    print("=" * 70)
    
    selected_attrs_1 = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"}
    ]
    
    context_1 = build_three_way_persona_context_v2(
        category, selected_attrs_1, user_id, processing_dir
    )
    print(context_1)
    
    # Test case 2: Query asks for something user dislikes
    print("\n\n" + "=" * 70)
    print("TEST 2: Query asks for Sizzix (user dislikes)")
    print("=" * 70)
    
    selected_attrs_2 = [
        {"dimension": "Brand_Preference", "value": "Sizzix"}
    ]
    
    context_2 = build_three_way_persona_context_v2(
        category, selected_attrs_2, user_id, processing_dir
    )
    print(context_2)
    
    # Test case 3: Multiple attributes
    print("\n\n" + "=" * 70)
    print("TEST 3: Multiple query attributes")
    print("=" * 70)
    
    selected_attrs_3 = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"}
    ]
    
    context_3 = build_three_way_persona_context_v2(
        category, selected_attrs_3, user_id, processing_dir
    )
    print(context_3)
