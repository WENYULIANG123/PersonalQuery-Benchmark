#!/usr/bin/env python3
"""
Three-way Preference Classifier for Stage 14 Reranking

Classifies historical user preferences into three categories:
1. Explicit Preferences: Positive attributes directly expressed by user
2. Implicit Preferences: Improvement expectations inferred from negative feedback
3. Conflicting Preferences: Contradictory signals within the same dimension

Used to build structured persona context for LLM reranking.
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict


class PreferenceClassifier:
    """
    Classifies user preferences into Explicit, Implicit, and Conflicting categories
    """
    
    def __init__(self, user_id: str, processing_dir: str):
        """
        Args:
            user_id: User ID
            processing_dir: Directory containing persona_*.json files from Stage 3
        """
        self.user_id = user_id
        self.processing_dir = processing_dir
        self.persona_cache = {}  # Cache loaded persona data by category
    
    def load_persona_data(self, category: str) -> List[Dict]:
        """
        Load persona attributes for a specific category
        
        Args:
            category: Product category (e.g., "Die-Cuts")
        
        Returns:
            List of attribute dictionaries
        """
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
    
    def classify_preferences_for_dimension(
        self, 
        dimension: str, 
        all_attributes: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Classify preferences for a single dimension into three categories
        
        Args:
            dimension: Dimension name (e.g., "Brand_Preference")
            all_attributes: All user attributes from persona data
        
        Returns:
            Dictionary with three keys:
            {
                'explicit': [positive attributes],
                'implicit': [negative attributes with improvement wishes],
                'conflicting': [attributes causing conflicts]
            }
        """
        # Filter attributes for this dimension
        dimension_attrs = [
            attr for attr in all_attributes 
            if attr.get('dimension') == dimension
        ]
        
        if not dimension_attrs:
            return {
                'explicit': [],
                'implicit': [],
                'conflicting': []
            }
        
        # Separate by sentiment
        positive_attrs = []
        negative_attrs = []
        neutral_attrs = []
        
        for attr in dimension_attrs:
            sentiment = attr.get('sentiment', 'neutral')
            if sentiment == 'positive':
                positive_attrs.append(attr)
            elif sentiment == 'negative':
                negative_attrs.append(attr)
            else:
                neutral_attrs.append(attr)
        
        # Classify
        explicit = []
        implicit = []
        conflicting = []
        
        # 1. Explicit: Positive attributes (clearly expressed preferences)
        explicit = positive_attrs.copy()
        
        # 2. Implicit: Negative attributes with improvement wishes
        #    These indicate what user doesn't like, implying what they DO want
        for attr in negative_attrs:
            improvement = attr.get('improvement_wish', '').strip()
            if improvement:  # Has improvement suggestion
                implicit.append(attr)
            else:
                # Negative without improvement is also implicit (avoid this)
                implicit.append(attr)
        
        # 3. Conflicting: Detect contradictions
        #    - Same dimension has both positive and negative sentiments
        #    - Same attribute value appears with different sentiments
        
        if positive_attrs and negative_attrs:
            # Dimension-level conflict: has both positive and negative preferences
            # Mark all as potentially conflicting
            
            # Check for attribute-level conflicts (same value, different sentiments)
            positive_values = {attr.get('attribute', '').lower() for attr in positive_attrs}
            negative_values = {attr.get('attribute', '').lower() for attr in negative_attrs}
            
            conflicting_values = positive_values & negative_values
            
            if conflicting_values:
                # Specific attribute conflict
                for attr in dimension_attrs:
                    if attr.get('attribute', '').lower() in conflicting_values:
                        conflicting.append(attr)
            else:
                # General dimension conflict (different attributes, contradictory signals)
                # Don't mark as conflicting unless there's a clear contradiction
                # This is handled by returning both explicit and implicit
                pass
        
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
        Classify preferences for all dimensions in a query's selected_attributes
        
        Args:
            category: Product category
            selected_attributes: List of {dimension, value} from query
        
        Returns:
            Dictionary mapping dimension -> {explicit, implicit, conflicting}
            Example:
            {
                'Brand_Preference': {
                    'explicit': [...],
                    'implicit': [...],
                    'conflicting': [...]
                },
                'Performance': {...}
            }
        """
        # Load persona data for this category
        all_attributes = self.load_persona_data(category)
        
        if not all_attributes:
            print(f"⚠️ No persona data found for category: {category}")
            return {}
        
        # Get unique dimensions from selected_attributes
        dimensions = list(set(attr.get('dimension') for attr in selected_attributes))
        
        # Classify preferences for each dimension
        result = {}
        for dimension in dimensions:
            if not dimension:
                continue
            
            classification = self.classify_preferences_for_dimension(
                dimension, 
                all_attributes
            )
            result[dimension] = classification
        
        return result
    
    def format_classified_preferences(
        self, 
        classified: Dict[str, Dict[str, List[Dict]]]
    ) -> str:
        """
        Format classified preferences into human-readable text for LLM context
        
        Args:
            classified: Output from classify_query_preferences
        
        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        
        for dimension, categories in classified.items():
            lines.append(f"\n### {dimension}")
            
            # Explicit preferences
            if categories['explicit']:
                lines.append("\n**Explicit Preferences (Positive):**")
                for attr in categories['explicit']:
                    value = attr.get('attribute', attr.get('value', ''))
                    evidence = attr.get('original_text', '')
                    lines.append(f"  ✓ {value}")
                    if evidence:
                        lines.append(f"    Evidence: \"{evidence[:100]}...\"")
            
            # Implicit preferences (from negative feedback)
            if categories['implicit']:
                lines.append("\n**Implicit Preferences (Inferred from Negative Feedback):**")
                for attr in categories['implicit']:
                    value = attr.get('attribute', attr.get('value', ''))
                    improvement = attr.get('improvement_wish', '')
                    lines.append(f"  ⚠ Dislikes: {value}")
                    if improvement:
                        lines.append(f"    → Expects: {improvement}")
            
            # Conflicting preferences
            if categories['conflicting']:
                lines.append("\n**Conflicting Preferences (Contradictory Signals):**")
                for attr in categories['conflicting']:
                    value = attr.get('attribute', attr.get('value', ''))
                    sentiment = attr.get('sentiment', 'neutral')
                    lines.append(f"  ⚔ {value} (sentiment: {sentiment})")
                lines.append("  ⚠️ **Note**: Query requirements take priority over conflicts")
        
        return "\n".join(lines) if lines else ""


def build_three_way_persona_context(
    category: str,
    selected_attributes: List[Dict],
    user_id: str,
    processing_dir: str
) -> str:
    """
    Build persona context with three-way preference classification
    
    Args:
        category: Product category
        selected_attributes: List of {dimension, value} from query
        user_id: User ID
        processing_dir: Directory containing persona files
    
    Returns:
        Formatted persona context string with classified preferences
    """
    if not selected_attributes:
        return ""
    
    classifier = PreferenceClassifier(user_id, processing_dir)
    classified = classifier.classify_query_preferences(category, selected_attributes)
    
    if not classified:
        return ""
    
    # Format for LLM
    header = "User Preference Profile (Three-way Classification):\n"
    header += "=" * 60
    body = classifier.format_classified_preferences(classified)
    
    return header + body if body else ""


# Backward compatibility alias
def classify_preferences(
    category: str,
    selected_attributes: List[Dict],
    user_id: str,
    processing_dir: str
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Simple function to classify preferences (returns raw data structure)
    """
    classifier = PreferenceClassifier(user_id, processing_dir)
    return classifier.classify_query_preferences(category, selected_attributes)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python preference_classifier.py <user_id> <category>")
        print("Example: python preference_classifier.py A13OFOB1394G31 Die-Cuts")
        sys.exit(1)
    
    user_id = sys.argv[1]
    category = sys.argv[2]
    processing_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/03_processing"
    
    # Example selected_attributes
    selected_attrs = [
        {"dimension": "Brand_Preference", "value": "Spellbinders"},
        {"dimension": "Performance", "value": "clean cutting"},
        {"dimension": "Product_Category", "value": "butterfly dies"}
    ]
    
    # Classify
    classifier = PreferenceClassifier(user_id, processing_dir)
    result = classifier.classify_query_preferences(category, selected_attrs)
    
    # Print formatted output
    print("\n" + "=" * 60)
    print(f"Preference Classification for {user_id} - {category}")
    print("=" * 60)
    
    formatted = classifier.format_classified_preferences(result)
    print(formatted)
    
    # Also print raw JSON
    print("\n" + "=" * 60)
    print("Raw JSON Output:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
