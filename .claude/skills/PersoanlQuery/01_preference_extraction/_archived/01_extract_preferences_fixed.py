#!/usr/bin/env python3
"""
Fixed version of Stage 1: Preference Extraction
Extract preferences from reviews using LLM - TARGET USER ONLY (other users commented out)

Changes from original:
1. Support both old (target_review) and new (target_reviews) data formats
2. Commented out all other users preference extraction
3. Updated statistics to only show target user data
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

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_fixed_dimension_prompt():
    """Get the fixed 21-dimension schema and output format"""

    dimensions_schema = """
### IMPORTANT: Use ONLY these 21 dimensions:

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
17. **Value** - Value for money (e.g., "good value", "great price", "worth the money")
18. **Packaging_Quantity** - Packaging/quantity (e.g., "generous amount", "value pack", "comes in set")

### Category 7: Other
19. **Brand_Preference** - Brand preference (e.g., "prefer X over Y", "only buy X brand", "dislike Z brand")
20. **Compatibility** - Compatibility requirements (e.g., "works with Cuttlebug", "fits A2 cards")
21. **Special_User_Needs** - Special user needs (e.g., "arthritis-friendly", "for left-handed")

### Dimension Definitions:
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
- Functionality: Product features/capabilities (e.g., "cuts paper", "creates embossed effect", "locks blade")
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
17. **Value** - Value for money (e.g., "good value", "great price", "worth the money")
18. **Packaging_Quantity** - Packaging/quantity (e.g., "generous amount", "value pack", "comes in set")

### Category 7: Other
19. **Brand_Preference** - Brand preference (e.g., "prefer X over Y", "only buy X brand", "dislike Z brand")
20. **Compatibility** - Compatibility requirements (e.g., "works with Cuttlebug", "fits A2 cards")
21. **Special_User_Needs** - Special user needs (e.g., "arthritis-friendly", "for left-handed")

**Product_Category vs Material_Composition:**
- Product_Category: Product type/name (e.g., "glitter glue", "embossing folder", "die cut", "scissors")
- Material_Composition: Raw material/ingredient ONLY (e.g., "plastic", "metal", "cotton", "leather", "silver-colored")
- WARNING: "silver glitter glue" → Product_Category (it's a product name, NOT material!)
- WARNING: "embossing folder material (plastic/metal)" → Product_Category (it's describing the product type)
- WARNING: Only extract raw materials like "plastic", "metal", "cotton" as Material_Composition

**Ease_of_Use vs Compatibility:**
- Ease_of_Use: How easy/convenient to use (e.g., "easy to use", "simple to assemble", "intuitive", "user-friendly")
- Compatibility: What systems/devices it works with (e.g., "works with Cuttlebug", "compatible with Sizzix", "fits A2 cards")
- WARNING: "compatible with X" or "works with X" → Compatibility (NOT Ease_of_Use!)
- WARNING: "easy to use with X" → Ease_of_Use describes ease, Compatibility describes the X

**Functionality vs Usage_Scenario:**
- Functionality: Product features/capabilities (e.g., "cuts paper", "creates embossed effect", "locks blade")
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
17. **Value** - Value for money (e.g., "good value", "great price", "worth the money")
18. **Packaging_Quantity** - Packaging/quantity (e.g., "generous amount", "value pack", "comes in set")

### Category 7: Other
19. **Brand_Preference** - Brand preference (e.g., "prefer X over Y", "only buy X brand", "dislike Z brand")
20. **Compatibility** - Compatibility requirements (e.g., "works with Cuttlebug", "fits A2 cards")
21. **Special_User_Needs** - Special user needs (e.g., "arthritis-friendly", "for left-handed")

IMPORTANT:
- Use EXACT dimension names as shown above (use underscore "_" instead of spaces)
- If no information for a dimension, use empty array []
- sentiment must be one of: positive, negative, neutral
- Extract specific evidence from the review text
"""

    output_format = """
Return a JSON object with this structure:
{
  "Product_Attributes": {
    "Product_Category": [...],
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

Each entity in the arrays must have this format:
{
  "entity": "specific preference",
  "sentiment": "positive|negative|neutral",
  "original_text": "exact quote from review",
  "improvement_wish": "what could be improved (empty string if none)"
}

IMPORTANT:
- Use EXACT dimension names as shown above (use underscore "_" instead of spaces)
- If no information for a dimension, use empty array []
- sentiment must be one of: positive, negative, neutral
- Extract specific evidence from the review text
"""

    return dimensions_schema, output_format


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
    """Process one product: extract preferences from target user reviews ONLY (other users commented out)"""
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

    # COMMENTED OUT: Not processing other users
    # other_reviews = product_data.get('other_reviews', [])

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

    # Extract target user preferences ONLY
    if target_review:
        target_prefs = extract_preferences_from_review(target_review, title, 'target')
        result['target_user_preferences'] = target_prefs

    # COMMENTED OUT: Other users preference extraction (not processing other users)
    # if other_reviews:
    #     all_other_prefs = {}
    #     for cat in fixed_categories:
    #         all_other_prefs[cat] = {}
    #
    #     for i, review in enumerate(other_reviews):
    #         try:
    #             prefs = extract_preferences_from_review(review, title, 'other')
    #             if prefs:
    #                 for category, category_data in prefs.items():
    #                     if category not in all_other_prefs:
    #                         all_other_prefs[category] = {}
    #                     if isinstance(category_data, dict):
    #                         for dimension, entities in category_data.items():
    #                             if dimension not in all_other_prefs[category]:
    #                                 all_other_prefs[category][dimension] = []
    #                             if isinstance(entities, list):
    #                                 all_other_prefs[category][dimension].extend(entities)
    #         except Exception as e:
    #             pass
    #
    #     result['other_users_preferences'] = all_other_prefs

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
    parser = argparse.ArgumentParser(description="Stage 1: Preference Extraction (TARGET USER ONLY)")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Number of products to process concurrently")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1: Preference Extraction (TARGET USER ONLY)")
    log_with_timestamp("=" * 80)
    log_with_timestamp("⚠️  MODIFIED: Only extracting target user preferences")
    log_with_timestamp("⚠️  MODIFIED: Supporting both old (target_review) and new (target_reviews) formats")

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
                    # Calculate dimension statistics (TARGET USER ONLY)
                    dimension_stats = defaultdict(lambda: {'target': 0, 'other': 0})

                    for r in results:
                        # Count target user preferences by dimension
                        for category, dims in r['target_user_preferences'].items():
                            if isinstance(dims, dict):
                                for dim, entities in dims.items():
                                    if isinstance(entities, list):
                                        dimension_stats[dim]['target'] += len(entities)

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
    log_with_timestamp("⚠️  NOTE: Only target user preferences extracted (other users skipped)")

if __name__ == "__main__":
    main()
