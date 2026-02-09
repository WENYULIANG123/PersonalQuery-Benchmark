#!/usr/bin/env python3
"""
Generate Match Prompts Script (User-Centric Version)
Implements Step 3.1 of the User Profile Manager workflow.
Logic:
1. Load consolidated user preferences (preferences_[USER_ID].json).
2. Load metadata for reviewed products.
3. Group products by category to find "neighbors" within the user's history.
4. For each product, verify "seeds" (positive/neutral preferences) against metadata.
5. Identify neighbor pain points (negative preferences) solved by the current product.
6. Generate reasoning prompts for the final Top 3 attribute selection.
"""

import json
import os
import sys
import argparse
import re
from typing import List, Dict, Set, Tuple
from datetime import datetime

# Simple stop words for keyword extraction
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in',
    'on', 'at', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
    'more', 'less', 'very', 'really', 'quite', 'too', 'much', 'good', 'bad'
}

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text for matching."""
    if not text:
        return set()
    words = re.findall(r'\b\w+\b', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}

def can_solve_pain_point(seeds_candidates: List[str], metadata_text: str, wish: str) -> bool:
    """Check if target product can solve neighbor's pain point."""
    wish_keywords = extract_keywords(wish)
    if not wish_keywords:
        return False

    # Check against user preferences (seeds)
    seeds_text = ' '.join(seeds_candidates).lower()
    for keyword in wish_keywords:
        if keyword in seeds_text:
            return True

    # Check against metadata
    metadata_lower = metadata_text.lower()
    for keyword in wish_keywords:
        if keyword in metadata_lower:
            return True

    return False

class MatchPromptGenerator:
    def __init__(self, input_file: str, meta_file: str):
        log_with_timestamp(f"Loading user preferences from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.user_id = self.data.get('user_id')
            self.results = self.data.get('results', [])
        
        self.meta_file = meta_file
        self.product_metadata = {} # Cache for metadata {asin: text_description}
        self._load_metadata()

    def _load_metadata(self):
        """Streaming metadata loader to save memory."""
        needed_asins = {item.get('asin') for item in self.results if item.get('asin')}
        log_with_timestamp(f"Loading metadata for {len(needed_asins)} products from {self.meta_file}")
        
        loaded_count = 0
        try:
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    # JSON list format
                    all_meta = json.load(f)
                    for item in all_meta:
                        asin = item.get('asin')
                        if asin in needed_asins:
                            self.product_metadata[asin] = self._format_metadata(item)
                            loaded_count += 1
                else:
                    # Line delimited JSON format
                    for line in f:
                        if not line.strip(): continue
                        try:
                            item = json.loads(line)
                            asin = item.get('asin')
                            if asin in needed_asins:
                                self.product_metadata[asin] = self._format_metadata(item)
                                loaded_count += 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log_with_timestamp(f"Error loading metadata: {e}")
            
        log_with_timestamp(f"Successfully loaded metadata for {loaded_count} products.")

    def _format_metadata(self, item: Dict) -> str:
        """Format raw metadata dictionary into a searchable string."""
        parts = []
        if item.get('title'): parts.append(f"Title: {item['title']}")
        
        features = item.get('feature', []) or item.get('features', [])
        if features:
            if isinstance(features, list):
                parts.append(f"Features: {' | '.join(features[:5])}")
            else:
                parts.append(f"Features: {str(features)[:300]}")
        
        desc = item.get('description', '')
        if desc:
            if isinstance(desc, list):
                parts.append(f"Description: {' '.join(desc)[:300]}")
            else:
                parts.append(f"Description: {str(desc)[:300]}")
                
        return " | ".join(parts)[:1500]

    def generate_prompts(self) -> List[Dict]:
        prompts = []
        
        # Group user's own products by category as local neighbors
        products_by_cat = {}
        for item in self.results:
            # Note: Category might be in preferences extraction or metadata
            # For now, we trust the preference extraction's category
            cat = item.get('preferences', {}).get('Product Category', 'Unknown')
            if cat not in products_by_cat:
                products_by_cat[cat] = []
            products_by_cat[cat].append(item)
            
        for item in self.results:
            target_asin = item.get('asin')
            target_title = item.get('product_title', 'Unknown')
            preferences = item.get('preferences', {})
            target_cat = preferences.get('Product Category', 'Unknown')
            
            metadata_text = self.product_metadata.get(target_asin, "Metadata not available")
            
            # Identify Positive Seeds
            seeds_candidates = []
            for category, entities in preferences.items():
                if category == "Product Category": continue
                if isinstance(entities, list):
                    for ent in entities:
                        if ent.get('sentiment') in ['positive', 'neutral']:
                            seeds_candidates.append(ent.get('entity', ''))
            
            # Identify Neighbor Insights from other products of the SAME user in SAME category
            neighbors = []
            possible_neighbors = products_by_cat.get(target_cat, [])
            for nb in possible_neighbors:
                if nb.get('asin') == target_asin: continue

                nb_prefs = nb.get('preferences', {})
                nb_wishes = []
                nb_positives = []

                for cat, ents in nb_prefs.items():
                    if cat == "Product Category": continue
                    if isinstance(ents, list):
                        for ent in ents:
                            if ent.get('sentiment') == 'negative' and ent.get('improvement_wish'):
                                nb_wishes.append(ent['improvement_wish'])
                            elif ent.get('sentiment') in ['positive', 'neutral']:
                                nb_positives.append(ent.get('entity', ''))

                # Pre-filter: only keep neighbor wishes that THIS product can solve
                matched_wishes = [wish for wish in nb_wishes if can_solve_pain_point(seeds_candidates, metadata_text, wish)]

                if matched_wishes or nb_positives:
                    neighbors.append({
                        "asin": nb.get('asin'),
                        "wishes": matched_wishes[:2],
                        "extras": nb_positives[:2]
                    })
            
            # Construct Final Task Prompt
            task_prompt = f"""**Preference Matching Task for User: {self.user_id} | ASIN: {target_asin}**

**1. Target Product Context**
*   **Title**: {target_title}
*   **Category**: {target_cat}
*   **Metadata**: {metadata_text}
*   **Extracted Preferences (Seeds)**: {json.dumps(seeds_candidates, ensure_ascii=False)}

**2. Neighbor Insights (Other products reviewed by this user in the same category)**
*   The following wishes from other products have been checked against your target product's capabilities:
{json.dumps(neighbors[:3], indent=2, ensure_ascii=False)}

**3. Your Mission (Step-by-Step Reasoning):**

*   **Step A: Verify Seeds (Self-Consistency)**
*   Compare "Extracted Preferences" with product **Metadata**.
*   Keep only those that are semantically supported by the title or features.

*   **Step B: Augment from Neighbor Pain Points**
*   Consider neighbor "wishes" (complaints about other products). 
*   If your target product solves a wish, add it as a "High Priority" attribute (e.g., neighbor complained "Too loud", your product is "Whisper Quiet").

*   **Step C: Final Selection (Top 3)**
*   Choose the 3 most representative attributes based on:
    1.  **Priority 1 (High)**: Resolved pain points (unique advantages over neighbors).
    2.  **Priority 2 (Medium)**: Very specific/unique features mentioned in metadata.
    3.  **Priority 3 (Low)**: Essential core specs (only if you have < 3 items).

**Output Format:**
```json
{{
  "target_asin": "{target_asin}",
  "selected_attributes": ["Attr1", "Attr2", "Attr3"],
  "category": "{target_cat}",
  "reasoning": "Explain your selection logic briefly here."
}}
```"""
            prompts.append({
                "user_id": self.user_id,
                "asin": target_asin,
                "prompt": task_prompt
            })
            
        return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate Step 3 Matching Prompts from User Preferences")
    parser.add_argument("--input", required=True, help="Path to preferences_[USER_ID].json")
    parser.add_argument("--meta-file", required=True, help="Path to raw product metadata file")
    parser.add_argument("--output-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/preference_match", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = MatchPromptGenerator(args.input, args.meta_file)
    prompts_data = generator.generate_prompts()
    
    output_file = os.path.join(args.output_dir, f"match_prompts_{generator.user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "user_id": generator.user_id,
            "total_products": len(prompts_data),
            "prompts": prompts_data
        }, f, indent=2, ensure_ascii=False)
        
    log_with_timestamp(f"✅ Generated {len(prompts_data)} match prompts for user {generator.user_id} -> {output_file}")

if __name__ == "__main__":
    main()
