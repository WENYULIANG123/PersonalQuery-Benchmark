#!/usr/bin/env python3
"""
Preference Match Script
Implements the logic to select attributes for query generation:
1. Validates Target Product Seeds (Verify Entity in Metadata)
2. Finds Neighbor Products (Same Category Strategy)
3. Selects Top 3 Attributes for the Match

Now supports loading Metadata directly from raw JSON file.
"""

import json
import os
import sys
import argparse
from typing import List, Dict, Set, Tuple
import re

# Simple stop words for keyword extraction
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in',
    'on', 'at', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
    'more', 'less', 'very', 'really', 'quite', 'too', 'much', 'good', 'bad'
}

def extract_keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords from text.
    Remove stop words and short words.
    """
    if not text:
        return set()

    # Convert to lowercase and extract words
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter out stop words and short words
    keywords = {w for w in words if w not in STOP_WORDS and len(w) > 2}

    return keywords

def can_solve_pain_point(seeds_candidates: List[str], metadata_text: str, wish: str) -> bool:
    """
    Check if target product can solve neighbor's pain point.

    Args:
        seeds_candidates: List of user's positive/neutral preferences
        metadata_text: Product metadata (title, features, etc.)
        wish: Neighbor's improvement_wish (positive version of complaint)

    Returns:
        True if target product mentions solving this pain point
    """
    wish_keywords = extract_keywords(wish)

    if not wish_keywords:
        return False

    # Check against user preferences
    seeds_text = ' '.join(seeds_candidates).lower()
    for keyword in wish_keywords:
        if keyword in seeds_text:
            return True  # Found in user preferences

    # Check against metadata
    metadata_lower = metadata_text.lower()
    for keyword in wish_keywords:
        if keyword in metadata_lower:
            return True  # Found in metadata

    return False

class PreferenceMatcher:
    def __init__(self, input_file: str, meta_file: str):
        with open(input_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                self.preferences = data
            else:
                self.preferences = [data]
        
        self.meta_file = meta_file
        self.product_metadata = {} # Cache for metadata {asin: text_description}
        self._load_metadata()

    def _load_metadata(self):
        """
        Load metadata ONLY for the products we are interested in (targets + neighbors).
        Streams the file to save memory.
        """
        print(f"Loading metadata from {self.meta_file}...")
        
        # 1. Identify all needed ASINs
        needed_asins = set()
        
        # Add target ASINs
        for item in self.preferences:
            if item.get('asin'):
                needed_asins.add(item.get('asin'))
                
        # We also need neighbor ASINs if we want to be thorough, 
        # but the current logic only checks TARGET metadata. 
        # Neighbors are found via 'preferences' list which has 'asin'.
        # Wait, neighbors are *other* items in the same `input_file`?
        # Yes, existing logic: `possible_neighbors = products_by_cat.get(target_cat, [])`
        # So 'needed_asins' is strictly all ASINs in `self.preferences`.
        
        print(f"Found {len(needed_asins)} unique ASINs to load metadata for.")
        
        # 2. Stream and filter
        loaded_count = 0
        try:
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                # The raw file is likely a list of JSON objects (standard JSON) 
                # OR line-delimited JSON. 
                # Based on previous extraction, it was `json.dump(data, f)` which creates a standard JSON list.
                # Standard JSON list is hard to stream without a parser like ijson, 
                # BUT the user said `raw/meta_...json` which WE extracted line-by-line 
                # OR is it the original raw file?
                # User's command: "从 raw/meta...json 得到". 
                # If it's the file I just extracted to `processed/meta...json` it is a JSON list.
                # If it's the `raw/...json.gz` (decompressed) implies line-delimited?
                # Let's check format. Amazon raw data is usually one JSON per line (pjson).
                # But wait, I ran `gzip -dk` on `meta_...json.gz`. 
                # Let's assume line-delimited for "raw" Amazon data usually.
                # However, if it's a JSON LIST (starts with [), simple line reading fails.
                
                # Let's act defensively. Read first char.
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    # It's a JSON list. Standard load. Hopefully fits in memory?
                    # The file is ~800MB. It MIGHT be tight but python usually manages.
                    print("Detected JSON list format. Loading entire file...")
                    all_meta = json.load(f)
                    for item in all_meta:
                        asin = item.get('asin')
                        if asin in needed_asins:
                            self.product_metadata[asin] = self._format_metadata(item)
                            loaded_count += 1
                else:
                    # Line delimited
                    print("Detected Line-delimited JSON format. Streaming...")
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
            print(f"Error loading metadata: {e}")
            
        print(f"Successfully loaded metadata for {loaded_count} products.")

    def _format_metadata(self, item: Dict) -> str:
        """From raw dictionary to formatted string"""
        parts = []
        
        # Title
        title = item.get('title', '')
        if title:
            parts.append(f"Title: {title}")
            
        # Features (often a list)
        features = item.get('feature', []) or item.get('features', [])
        if features:
            if isinstance(features, list):
                parts.append(f"Features: {' | '.join(features[:5])}") # Top 5 features
            else:
                parts.append(f"Features: {str(features)[:300]}...")
        
        # Description (often text or list)
        desc = item.get('description', '')
        if desc:
            if isinstance(desc, list):
                parts.append(f"Description: {' '.join(desc)[:300]}...")
            else:
                parts.append(f"Description: {str(desc)[:300]}...")
                
        return " | ".join(parts)[:1500] # Cap total length

    def get_product_metadata_text(self, asin: str) -> str:
        return self.product_metadata.get(asin, "Metadata not available")

    def generate_full_prompts(self) -> List[Dict]:
        """
        Generate a comprehensive prompt for the Agent to perform the full logic:
        1. Verify Seeds
        2. Check Neighbors (from the provided list)
        3. Select Top 3
        """
        prompts = []
        
        # 1. Group by Category to help find neighbors easily
        products_by_cat = {}
        for item in self.preferences:
            cat = item.get('extraction', {}).get('Product Category', 'Unknown')
            if cat not in products_by_cat:
                products_by_cat[cat] = []
            products_by_cat[cat].append(item)
            
        for item in self.preferences:
            target_asin = item.get('asin')
            target_cat = item.get('extraction', {}).get('Product Category', 'Unknown')
            extraction = item.get('extraction', {})
            
            # Metadata
            metadata_text = self.get_product_metadata_text(target_asin)
            
            # Seeds Candidates
            seeds_candidates = []
            for category, entities in extraction.items():
                if category == "Product Category": continue
                if isinstance(entities, list):
                    for ent in entities:
                        if ent.get('sentiment') in ['positive', 'neutral']:
                            seeds_candidates.append(ent.get('entity', ''))
            
            # Neighbors Candidates (Same Category)
            neighbors = []
            possible_neighbors = products_by_cat.get(target_cat, [])
            for nb in possible_neighbors:
                if nb.get('asin') == target_asin: continue

                # Extract neighbor's data
                nb_extraction = nb.get('extraction', {})
                nb_all_wishes = [] 
                nb_extras = []

                for cat, ents in nb_extraction.items():
                    if cat == "Product Category": continue
                    if isinstance(ents, list):
                        for ent in ents:
                            if ent.get('sentiment') == 'negative':
                                if ent.get('improvement_wish'):
                                    nb_all_wishes.append(ent['improvement_wish'])
                            elif ent.get('sentiment') in ['positive', 'neutral']:
                                nb_extras.append(ent.get('entity', ''))

                # Filter wishes: only keep those target product can solve
                matched_wishes = []
                for wish in nb_all_wishes:
                    if can_solve_pain_point(seeds_candidates, metadata_text, wish):
                        matched_wishes.append(wish)

                # Only add neighbor if they have relevant insights
                if matched_wishes or nb_extras:
                    neighbors.append({
                        "asin": nb.get('asin'),
                        "wishes": matched_wishes[:2],  # Only matched wishes
                        "extras": nb_extras[:2]
                    })
            
            # Construct Prompt
            task_prompt = f"""**Preference Matching Task for ASIN: {target_asin}**

**1. Target Product Context**
*   **Category**: {target_cat}
*   **Metadata**: {metadata_text}
*   **User Preferences (Seeds)**: {json.dumps(seeds_candidates, ensure_ascii=False)}

**2. Neighbor Insights (Pre-filtered - Only Matched Wishes)**
*   The following neighbors' wishes have been **verified to match** your target product's attributes or metadata:
{json.dumps(neighbors[:3], indent=2, ensure_ascii=False)}
*(Note: These wishes are already filtered - only showing ones that match your product's capabilities)*

**3. Your Mission (Step-by-Step Reasoning):**

*   **Step A: Verify Seeds (Self-Consistency)**
*   Check each "User Preference Seed". Does it match the **Metadata** semantics?
*   Keep only the VALID ones.

*   **Step B: Augment from Neighbors (Pain Points)**
*   Each neighbor "wish" below has been pre-validated: your target product mentions solving this pain point (in user preferences OR metadata).
*   **Add the wish as a Positive Attribute** if it provides a differentiating advantage.
*   *Example*: Neighbor complained "Too flimsy", your product has "Reinforced Steel" → Add "Sturdy construction" as Priority 1.

*   **Step C: Final Selection (The "Match")**
*   Combine Valid Seeds + Augmented Neighbor Wishes.
*   **Select the TOP 3 most important attributes** based on this priority:
    1.  **Priority 1 (High): Resolved Pain Points**. (e.g., Neighbor complained "Too flimsy", your product solves it → Strong differentiator).
    2.  **Priority 2 (Medium): Unique/Specific Features**. (e.g., "Waterproof", "Noise-cancelling", "Pearlescent"). Prefer specific terms over generic ones.
    3.  **Priority 3 (Low): Core Specs/Functionality**. (e.g., "Wireless", "X-Large"). Essential but less differentiating.
*   **Constraint**: If you have >3 candidates, **DROP the most generic ones (Priority 3) first**. Keep the Differentiators (Priority 1 & 2).

**Output Format:**
```json
{{
  "target_asin": "{target_asin}",
  "selected_attributes": ["Attr1", "Attr2", "Attr3"],
  "reasoning": "Selected 'Sturdy construction' (Priority 1 - Resolves neighbor's 'too flimsy' complaint, verified in metadata). Selected 'Waterproof' (Priority 2 - Unique feature). Dropped 'Good Quality' (Priority 3 - Generic)."
}}
```
"""
            prompts.append({
                "asin": target_asin,
                "prompt": task_prompt
            })
            
        return prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to final_preferences.json")
    parser.add_argument("--meta_file", required=True, help="Path to raw metadata json file")
    parser.add_argument("--user_id", default="A13OFOB1394G31", help="Filter for specific user ID (optional)")
    parser.add_argument("--output", required=True, help="Path to save generated matching prompts")
    args = parser.parse_args()
    
    matcher = PreferenceMatcher(args.input, args.meta_file)
    prompts = matcher.generate_full_prompts()
    
    with open(args.output, 'w') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Generated preference matching prompts for {len(prompts)} products.")

if __name__ == "__main__":
    main()
