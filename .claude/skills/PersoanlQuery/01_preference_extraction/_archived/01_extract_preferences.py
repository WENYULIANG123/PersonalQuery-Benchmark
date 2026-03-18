#!/usr/bin/env python3
"""
Stage 1: Preference Extraction
Extract preferences from reviews using LLM - separate target and other users

Input: reviews_{USER_ID}.json from Stage 0
Output: preferences_{USER_ID}.json with extracted preferences
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient

# ============================================================================
# SEMANTIC DEDUPLICATION - COMMENTED OUT
# ============================================================================
# # Try to import sentence-transformers for semantic similarity
# try:
#     from sentence_transformers import SentenceTransformer
#     from sentence_transformers import util as st_util
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
#     st_util = None
#     SentenceTransformer = None
# 
# # Global model cache
# _semantic_model = None
# 
# def get_semantic_model():
#     """Get or load the semantic similarity model"""
#     global _semantic_model
#     if _semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
#         try:
#             _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # type: ignore
#         except Exception as e:
#             log_with_timestamp(f"Warning: Failed to load semantic model: {e}")
#             return None
#     return _semantic_model
# 
# 
# def check_and_fix_dimension_classification(prefs: Dict, similarity_threshold: float = 0.70) -> Dict:
#     """
#     Check and fix dimension classification for extracted preferences.
#     
#     For each product's target user preferences:
#     - If two attributes are similar (>= threshold) and in the SAME dimension: merge them
#     - If two attributes are similar (>= threshold) but in DIFFERENT dimensions: ask LLM to re-classify
#     
#     Returns the fixed preferences.
#     """
#     if not SENTENCE_TRANSFORMERS_AVAILABLE or st_util is None:
#         return prefs
#     
#     model = get_semantic_model()
#     if model is None:
#         return prefs
#     
#     all_attrs = []
#     for category, category_data in prefs.items():
#         if not isinstance(category_data, dict):
#             continue
#         for dimension, entities in category_data.items():
#             if not isinstance(entities, list):
#                 continue
#             for entity in entities:
#                 if not isinstance(entity, dict):
#                     continue
#                 entity_text = entity.get('entity', '')
#                 if entity_text:
#                     all_attrs.append({
#                         'dimension': dimension,
#                         'entity': entity_text,
#                         'sentiment': entity.get('sentiment', 'neutral'),
#                         'original_text': entity.get('original_text', ''),
#                         'improvement_wish': entity.get('improvement_wish', '')
#                     })
#     
#     if len(all_attrs) < 2:
#         return prefs
#     
#     attr_strings = [a['entity'] for a in all_attrs]
#     embeddings = model.encode(attr_strings, batch_size=32, show_progress_bar=False, convert_to_tensor=True)
#     cos_scores = st_util.cos_sim(embeddings, embeddings)
#     
#     to_remove = set()
#     merge_map = {}
#     same_dim_pairs = []
#     cross_dim_pairs = []
#     
#     for i in range(len(all_attrs)):
#         if i in to_remove or i in merge_map:
#             continue
#         for j in range(i + 1, len(all_attrs)):
#             if j in to_remove or j in merge_map:
#                 continue
#             
#             dim_i = all_attrs[i]['dimension']
#             dim_j = all_attrs[j]['dimension']
#             score = cos_scores[i][j].item()
#             
#             if score >= similarity_threshold:
#                 if dim_i == dim_j:
#                     same_dim_pairs.append((i, j, score))
#                 else:
#                     cross_dim_pairs.append((i, j, dim_i, dim_j, score))
#     
#     for i, j, score in same_dim_pairs:
#         if j not in to_remove:
#             to_remove.add(j)
#             merge_map[j] = i
#     
#     identical_cross_dim = []
#     other_cross_dim = []
#     for pair in cross_dim_pairs:
#         i, j, dim_i, dim_j, score = pair
#         entity_i = all_attrs[i]['entity'].lower().strip()
#         entity_j = all_attrs[j]['entity'].lower().strip()
#         if entity_i == entity_j:
#             identical_cross_dim.append(pair)
#         else:
#             other_cross_dim.append(pair)
#     
#     if cross_dim_pairs:
#         log_with_timestamp(f"    [Dimension Check] Found {len(cross_dim_pairs)} cross-dimension pairs ({len(identical_cross_dim)} identical, {len(other_cross_dim)} similar)")
#     
#     for i, j, dim_i, dim_j, score in identical_cross_dim:
#         if j not in to_remove:
#             to_remove.add(j)
#             merge_map[j] = i
#             log_with_timestamp(f"    [Dimension Check] Merged identical: '{all_attrs[i]['entity']}' in [{dim_i}] vs [{dim_j}]")
#     
#     for i, j, dim_i, dim_j, score in other_cross_dim:
#         if i in to_remove or j in to_remove:
#             continue
#         
#         entity_i = all_attrs[i]['entity']
#         entity_j = all_attrs[j]['entity']
#         
#         if len(entity_i) >= len(entity_j):
#             to_remove.add(j)
#             merge_map[j] = i
#             log_with_timestamp(f"    [Dimension Check] Kept longer: '{entity_i}' [{dim_i}] over '{entity_j}' [{dim_j}]")
#         else:
#             to_remove.add(i)
#             merge_map[i] = j
#             log_with_timestamp(f"    [Dimension Check] Kept longer: '{entity_j}' [{dim_j}] over '{entity_i}' [{dim_i}]")
#     
#     remaining_attrs = [all_attrs[k] for k in range(len(all_attrs)) if k not in to_remove]
#     
#     new_category_data = defaultdict(list)
#     for attr in remaining_attrs:
#         new_category_data[attr['dimension']].append({
#             'entity': attr['entity'],
#             'sentiment': attr['sentiment'],
#             'original_text': attr['original_text'],
#             'improvement_wish': attr['improvement_wish']
#         })
#     
#     for category, category_data in prefs.items():
#         if not isinstance(category_data, dict):
#             continue
#         for dim in list(category_data.keys()):
#             if dim in new_category_data:
#                 category_data[dim] = new_category_data[dim]
#             else:
#                 category_data[dim] = []
#     
#     return prefs
# 
# 
# def ask_llm_for_dimension(entity1: str, entity2: str, dim1: str, dim2: str) -> str:
#     """
#     Ask LLM to decide which dimension is correct for each entity.
#     Returns: dim1, dim2, or "keep_both"
#     """
#     prompt = f"""Two attributes were extracted from the same review but placed in different dimensions.
# However, they have very similar meaning (semantic similarity >= 0.70).
# 
# Attribute 1: "{entity1}" in dimension [{dim1}]
# Attribute 2: "{entity2}" in dimension [{dim2}]
# 
# Based on the dimension definitions below, decide which dimension is CORRECT for each attribute:
# 
# ### Dimension Definitions:
# - Product_Category: Product type/name (e.g., "glitter glue", "embossing folder", "die cut")
# - Material_Composition: Raw material ONLY (e.g., "plastic", "metal", "cotton")
# - Functionality: What the product does/features (e.g., "cuts paper", "creates patterns")
# - Performance: How well it performs (e.g., "works well", "cleans very well")
# - Ease_of_Use: How easy to use (e.g., "easy to assemble", "simple to use")
# - Compatibility: What devices it works with (e.g., "works with Cuttlebug", "compatible with Sizzix")
# - Size_Dimensions: Size specifications (e.g., "A2 size", "large", "compact")
# - Style_Design: Visual style (e.g., "modern", "intricate pattern", "decorative")
# - Usage_Scenario: Where/how to use (e.g., "for greeting cards", "at home")
# - Target_User: Intended user (e.g., "for beginners", "card makers")
# - Special_Purpose: Special use case (e.g., "for charity", "for crafting")
# 
# Output format:
# {{
#   "entity1_dimension": "correct dimension for entity1",
#   "entity2_dimension": "correct dimension for entity2",
#   "reason": "brief explanation"
# }}
# 
# Output ONLY valid JSON."""
# 
#     try:
#         client = LLMClient()
#         response = client.call(prompt, max_tokens=512)
#         
#         import re
#         match = re.search(r'\{.*\}', response, re.DOTALL)
#         if match:
#             result = json.loads(match.group(0))
#             entity1_dim = result.get('entity1_dimension', '')
#             entity2_dim = result.get('entity2_dimension', '')
#             
#             if entity1_dim == entity2_dim:
#                 if entity1_dim == dim1:
#                     return dim1
#                 elif entity1_dim == dim2:
#                     return dim2
#             else:
#                 if entity1_dim == dim1 and entity2_dim == dim2:
#                     return "keep_both"
#                 elif entity1_dim == dim1:
#                     return dim1
#                 elif entity2_dim == dim2:
#                     return dim2
#     except Exception as e:
#         log_with_timestamp(f"    [Dimension Check] LLM error: {e}")
#     
#     return "keep_both"
# ============================================================================
# END OF COMMENTED OUT SEMANTIC DEDUPLICATION
# ============================================================================

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def parse_response(response: str) -> Dict:
    """Parse JSON from LLM response"""
    try:
        import re
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        else:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except:
        pass
    return None

def get_fixed_dimension_prompt():
    """Return the fixed 21-dimension schema prompt (English only, no shopping behavior)"""
    
    # Fixed dimension schema: 7 categories, 21 dimensions
    dimensions_schema = """
## Fixed Dimension Schema (21 Dimensions, 7 Categories)

You MUST extract preferences into these FIXED dimensions ONLY. Do NOT create new dimensions.

### CRITICAL DIMENSION BOUNDARIES - Read Carefully

**Product_Category vs Material_Composition:**
- Product_Category: Product type/name (e.g., "glitter glue", "embossing folder", "die cut", "scissors")
- Material_Composition: Raw material/ingredient (e.g., "plastic", "metal", "cotton", "leather", "silver-colored")
- WARNING: "silver glitter glue" → Product_Category (it's a product name, NOT material!)
- WARNING: "embossing folder material (plastic/metal)" → Product_Category (it's describing the product type)
- WARNING: Only extract raw materials like "plastic", "metal", "cotton" as Material_Composition

**Ease_of_Use vs Compatibility:**
- Ease_of_Use: How easy/convenient to use (e.g., "easy to use", "simple to assemble", "intuitive", "user-friendly")
- Compatibility: What systems/devices it works with (e.g., "works with Cuttlebug", "compatible with Sizzix", "fits A2 cards")
- WARNING: "compatible with X" or "works with X" → Compatibility (NOT Ease_of_Use!)
- WARNING: "easy to use with X" → Ease_of_Use describes ease, Compatibility describes the X

**Functionality vs Usage_Scenario:**
- Functionality: What the product does/features (e.g., "cuts paper", "creates embossed effect", "locks blade")
- Usage_Scenario: Where/how the user uses it (e.g., "for greeting cards", "at home", "for scrapbooking")
- WARNING: "cutting cardstock" → Functionality (what product can do)
- WARNING: "making greeting cards at home" → Usage_Scenario (user's context)

**Functionality vs Performance:**
- Functionality: Product has what features (e.g., "has auto-off", "can cut thick materials")
- Performance: How well it performs (e.g., "cuts cleanly", "works great", "excellent performance")

### Category 1: Product Attributes
1. **Product_Category** - Product type/name (e.g., "glitter glue", "embossing folder", "scissors")
2. **Functionality** - Product features/capabilities (e.g., "trap dust", "cuts paper", "creates patterns")
3. **Material_Composition** - Raw material/ingredient ONLY (e.g., "cotton", "plastic", "toxic-free")

### Category 2: Quality Attributes
4. **Quality_Craftsmanship** - Quality/workmanship (e.g., "high quality", "well-made", "sturdy")
5. **Performance** - Performance effectiveness (e.g., "works well", "cleans very well", "cuts cleanly")
6. **Safety** - Safety requirements (e.g., "safe ingredients", "non-magnetic")

### Category 3: Appearance/Design
7. **Appearance_Color** - Visual appearance (e.g., "beautiful color", "cute", "attractive", "silver")
8. **Size_Dimensions** - Size fit (e.g., "perfect size", "fit well", "compact", "A2 size")
9. **Style_Design** - Style preference (e.g., "flowy", "modern", "two-toned", "intricate pattern")

### Category 4: User Experience
10. **Comfort** - Comfort level (e.g., "comfortable", "soft", "squishy")
11. **Ease_of_Use** - Usability ONLY (e.g., "easy to assemble", "easy to clean", "simple to use")
12. **Portability** - Portability (e.g., "portable", "lightweight", "foldable")

### Category 5: Usage Scenarios
13. **Target_User** - Intended user (e.g., "for my children", "for beginners", "card makers")
14. **Usage_Scenario** - Where/how to use (e.g., "outdoor", "for travel", "at home", "for greeting cards")
15. **Special_Purpose** - Special use case (e.g., "for Halloween", "for emergency", "charity crafting")

### Category 6: Price/Value
16. **Price** - Price related (e.g., "affordable", "expensive", "pricey")
17. **Value** - Value for money (e.g., "worth every penny", "bang for buck", "poor value")
18. **Packaging_Quantity** - Packaging specs (e.g., "value pack of 4", "set", "single bottle")

### Category 7: Special Requirements
19. **Compatibility** - Device/system compatibility (e.g., "works with Cuttlebug", "compatible with Sizzix")
20. **Special_User_Needs** - Special user requirements (e.g., "sensitive skin", "eczema")
21. **Brand_Preference** - Brand preference (e.g., "Fiskars", "Otterbox", "name brand")

**CRITICAL: Brand_Preference vs Compatibility Distinction:**
- "Sizzix", "Cottage Cutz", "Spellbinders" ALONE → Brand_Preference (user likes these brands)
- "works with Sizzix", "compatible with Cuttlebug", "fits Spellbinders" → Compatibility (machine compatibility)
- "Sizzix Grand Calibur" → Brand_Preference (the product IS the brand)
- "Movers & Shapers base dies" → Brand_Preference (it's a product/brand name)
- "requires Movers & Shapers trays" → Compatibility (compatibility requirement)

### Few-Shot Examples

Example 1:
Review: "I love this silver glitter glue for my scrapbooking projects. It works beautifully with my Cuttlebug machine."
Correct extraction:
- Product_Category: ["glitter glue"] (NOT Material_Composition - it's a product name!)
- Compatibility: ["works with Cuttlebug machine"] (NOT Ease_of_Use!)

Example 2:
Review: "This embossing folder creates beautiful raised patterns on cardstock. Easy to use with my die-cutting machine."
Correct extraction:
- Product_Category: ["embossing folder"]
- Functionality: ["creates raised pattern on cardstock"]
- Ease_of_Use: ["easy to use"]
- Compatibility: ["works with die-cutting machine"]

Example 3:
Review: "The blade has excellent cutting performance on heavy paper. Makes cutting long lengths very easy."
Correct extraction:
- Functionality: ["cuts heavy paper"] (what it can do)
- Performance: ["excellent cutting performance"] (how well it does)
- Ease_of_Use: ["easy to cut long lengths"] (usability)

Example 4:
Review: "This plastic embossing folder is perfect for making greeting cards at home."
Correct extraction:
- Product_Category: ["embossing folder"]
- Material_Composition: ["plastic"] (raw material)
- Usage_Scenario: ["making greeting cards at home"]

Example 5:
Review: "I love my Sizzix Grand Calibur die and all my Cottage Cutz dies. They work great with my Cuttlebug machine."
Correct extraction:
- Brand_Preference: ["Sizzix", "Cottage Cutz"] (user likes these brands)
- Compatibility: ["works with Cuttlebug machine"] (machine compatibility)
- Product_Category: ["die"]
NOTE: "Sizzix Grand Calibur" is a product name that includes brand → Brand_Preference
      "Cottage Cutz" is a brand name → Brand_Preference
      "works with Cuttlebug" describes compatibility → Compatibility

Example 6:
Review: "This die requires Movers & Shapers base trays to work properly."
Correct extraction:
- Compatibility: ["requires Movers & Shapers base trays"] (compatibility requirement)
NOTE: When a brand name appears with "requires", "works with", "compatible" → Compatibility
      When a brand name appears alone or with "prefer", "like", "love" → Brand_Preference
"""
    
    output_format = """
## Output Format
You MUST output JSON with this exact structure:

{
  "Product_Attributes": {
    "Product_Category": [{"entity": "...", "sentiment": "positive/negative/neutral", "original_text": "...", "improvement_wish": ""}],
    "Functionality": [...],
    "Material_Composition": [...]
  },
  "Quality_Attributes": {
    "Quality_Craftsmanship": [...],
    "Performance": [...],
    "Safety": [...]
  },
  "Appearance_Design": {
    "Appearance_Color": [...],
    "Size_Dimensions": [...],
    "Style_Design": [...]
  },
  "User_Experience": {
    "Comfort": [...],
    "Ease_of_Use": [...],
    "Portability": [...]
  },
  "Usage_Scenarios": {
    "Target_User": [...],
    "Usage_Scenario": [...],
    "Special_Purpose": [...]
  },
  "Price_Value": {
    "Price": [...],
    "Value": [...],
    "Packaging_Quantity": [...]
  },
  "Special_Requirements": {
    "Compatibility": [...],
    "Special_User_Needs": [...],
    "Brand_Preference": [...]
  }
}

IMPORTANT: 
- Use EXACT dimension names as shown above (use underscore "_" instead of spaces)
- If no information for a dimension, use empty array []
- sentiment must be one of: positive, negative, neutral
- Extract specific evidence from the review text
"""
    
    return dimensions_schema, output_format


def extract_preferences_from_review(review, product_title: str, user_type: str) -> Dict:
    """Extract preferences from a single review using fixed 21-dimension schema"""
    client = LLMClient()
    
    # Handle both string and dict formats (new vs old data format)
    if isinstance(review, str):
        # New format: review is just a string
        reviewer_id = ''
        review_text = review
        rating = 0
    else:
        # Old format: review is a dict
        reviewer_id = review.get('reviewerID', '')
        review_text = review.get('reviewText', '')
        rating = review.get('overall', 0)

    dimensions_schema, output_format = get_fixed_dimension_prompt()
    
    prompt = f"""Extract user preferences from this product review using the FIXED 21-dimension schema.

**Product**: {product_title}
**Rating**: {rating}/5
**Review**: {review_text}

{dimensions_schema}
{output_format}

Extract now. Output ONLY valid JSON, no explanation."""

    try:
        response = client.call(prompt, max_tokens=2048)
        prefs = parse_response(response)

        if prefs:
            # Mark user type and reviewer ID
            for category, category_data in prefs.items():
                if not isinstance(category_data, dict):
                    continue
                for dimension, entities in category_data.items():
                    if not isinstance(entities, list):
                        continue
                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        entity['reviewer_id'] = reviewer_id
                        entity['user_type'] = user_type
        return prefs or {}
    except Exception as e:
        log_with_timestamp(f"Error extracting from {reviewer_id}: {e}")
        return {}

def process_product(product_data: Dict) -> Dict:
    """Process one product: extract preferences from target user reviews only (other users commented out)"""
    asin = product_data['asin']
    title = product_data['product_title']

    # ADAPTED TO SUPPORT BOTH OLD AND NEW DATA FORMATS
    # Old format: target_review (single string)
    # New format: target_reviews (array of strings)
    target_review = product_data.get('target_review')
    if not target_review:
        # Try new format: get first element from target_reviews array
        target_reviews = product_data.get('target_reviews', [])
        if target_reviews and len(target_reviews) > 0:
            target_review = target_reviews[0]
        else:
            target_review = None

    # other_reviews = product_data.get('other_reviews', [])  # COMMENTED OUT: Not processing other users

    # Fixed dimension categories (English only)
    fixed_categories = [
        'Product_Attributes', 'Quality_Attributes', 'Appearance_Design',
        'User_Experience', 'Usage_Scenarios', 'Price_Value', 'Special_Requirements'
    ]

    result = {
        'asin': asin,
        'product_title': title,
        'target_user_id': product_data['target_user_id'],
        'target_user_preferences': {},
        'other_users_preferences': {}  # Will remain empty as we're not processing other users
    }

    # Initialize target preferences with fixed structure
    for cat in fixed_categories:
        result['target_user_preferences'][cat] = {}

    # Extract target user preferences
    if target_review:
        target_prefs = extract_preferences_from_review(target_review, title, 'target')
        # SEMANTIC DEDUPLICATION COMMENTED OUT
        # target_prefs = check_and_fix_dimension_classification(target_prefs)
        result['target_user_preferences'] = target_prefs

    # Extract other users' preferences (COMMENTED OUT - Only processing target user)
    # if other_reviews:
    #     all_other_prefs = {}
    #     # Initialize with fixed structure
    #     for cat in fixed_categories:
    #         all_other_prefs[cat] = {}
    #
    #     for i, review in enumerate(other_reviews):
    #         try:
    #             prefs = extract_preferences_from_review(review, title, 'other')
    #             if prefs:
    #                 # Handle nested preference categories
    #                 for category, category_data in prefs.items():
    #                     if category not in all_other_prefs:
    #                         all_other_prefs[category] = {}
    #                     if isinstance(category_data, dict):
    #                         for dimension, entities in category_data.items():
    #                             if dimension not in all_other_prefs[category]:
    #                                 all_other_prefs[category][dimension] = []
    # COMMENTED OUT: Other users preference extraction (only processing target user)
    #                             if isinstance(entities, list):
    #                                 all_other_prefs[category][dimension].extend(entities)
    #             except Exception as e:
    #                 pass  # Silent fail for individual reviews
    #
    #         result['other_users_preferences'] = all_other_prefs

    # Calculate statistics - handle nested structure
    def count_entities(prefs_dict):
        """Count entities in nested preference dict"""
        total = 0
        for category, category_data in prefs_dict.items():
            if isinstance(category_data, dict):
                for dimension, entities in category_data.items():
                    if isinstance(entities, list):
                        total += len(entities)
            elif isinstance(category_data, list):
                total += len(category_data)
        return total

    def count_categories(prefs_dict):
        """Count non-empty categories"""
        count = 0
        for category, category_data in prefs_dict.items():
            if isinstance(category_data, dict):
                if any(isinstance(v, list) and len(v) > 0 for v in category_data.values()):
                    count += 1
            elif isinstance(category_data, list) and len(category_data) > 0:
                count += 1
        return count

    target_count = count_entities(result['target_user_preferences'])
    other_count = 0  # Not processing other users, so always 0

    result['preference_breakdown'] = {
        'target_user': {
            'categories': count_categories(result['target_user_preferences']),
            'entities': target_count
        },
        'other_users': {
            'categories': 0,  # Not processing other users
            'entities': 0  # Not processing other users
        }
    }

    return result

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preference Extraction")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=50, 
                        help="Number of products to process concurrently")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1: Preference Extraction (Product-Level Concurrency)")
    log_with_timestamp("=" * 80)

    # Load Stage 0 output
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    user_id = data['user_id']
    products = data['results']

    log_with_timestamp(f"User: {user_id}")
    log_with_timestamp(f"Products: {len(products)}")
    log_with_timestamp(f"Concurrency: {args.max_workers} products in parallel")

    # Process all products WITH CONCURRENCY at product level
    results = []
    completed_count = [0]  # Use list for mutable counter in closure

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_product = {
            executor.submit(process_product, product): product
            for product in products
        }

        for future in as_completed(future_to_product):
            product = future_to_product[future]
            asin = product['asin']
            try:
                result = future.result()
                results.append(result)
                completed_count[0] += 1

                # Log progress every 10 products with dimension statistics
                if completed_count[0] % 10 == 0 or completed_count[0] == len(products):
                    # Calculate dimension statistics (TARGET USER ONLY - other users processing disabled)
                    dimension_stats = defaultdict(lambda: {'target': 0, 'other': 0})

                    for r in results:
                        # Count target user preferences by dimension
                        for category, dims in r['target_user_preferences'].items():
                            if isinstance(dims, dict):
                                for dim, entities in dims.items():
                                    if isinstance(entities, list):
                                        dimension_stats[dim]['target'] += len(entities)

                        # COMMENTED OUT: Count other users preferences by dimension (not processing other users)
                        # for category, dims in r['other_users_preferences'].items():
                        #     if isinstance(dims, dict):
                        #         for dim, entities in dims.items():
                        #             if isinstance(entities, list):
                        #                 dimension_stats[dim]['other'] += len(entities)

                    log_with_timestamp(f"Progress: {completed_count[0]}/{len(products)} products completed")
                    log_with_timestamp("  Preference extraction by dimension:")
                    log_with_timestamp(f"  {'Dimension':<30} {'Target':>10} {'Other':>10} {'Total':>10}")
                    log_with_timestamp("  " + "-" * 65)

                    # Sort dimensions by target count (only processing target user)
                    sorted_dims = sorted(dimension_stats.items(),
                                       key=lambda x: x[1]['target'],
                                       reverse=True)

                    for dim, counts in sorted_dims:
                        total = counts['target']  # Only target, since we're not processing other users
                        log_with_timestamp(f"  {dim:<30} {counts['target']:>10} {counts['other']:>10} {total:>10}")

                    total_prefs = sum(c['target'] for c in dimension_stats.values())  # Only target
                    log_with_timestamp(f"  {'TOTAL':<30} {total_prefs:>10} {0:>10} {total_prefs:>10}")
                    log_with_timestamp("")
            except Exception as e:
                log_with_timestamp(f"Error processing {asin}: {e}")
                completed_count[0] += 1

    log_with_timestamp(f"Completed: {len(results)}/{len(products)} products")

    # Save results
    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'total_products': len(results),
        'results': results
    }

    output_file = os.path.join(args.output_dir, f'preferences_{user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nSaved to {output_file}")
    log_with_timestamp("Stage 1 Complete!")

if __name__ == "__main__":
    main()
