#!/usr/bin/env python3
"""
Stage 1 Post-Processing: Fix Misclassified Preferences
Apply rules to correct dimension classifications based on semantic patterns.

Total correction rules: 83
- Initial rules: 67
- First update: +13 = 80 (Functionality/Usage/Brand/Compat boundaries)
- Second update: +3 = 83 (Quality_Craftsmanship/Value boundaries)

Last updated: 2026-03-12
"""
import os
import json
import sys
import re
from datetime import datetime
from typing import Dict, List, Any

# Correction rules: (pattern, source_dimension, target_dimension, reason)
# Pattern is a keyword/phrase that indicates the correct classification
CORRECTION_RULES = [
    # Comfort vs Size_Dimensions
    ("fits perfectly in hand", "Size_Dimensions", "Comfort", "describes physical comfort, not size"),
    ("fits hand", "Size_Dimensions", "Comfort", "describes physical comfort"),
    ("comfortable", "Size_Dimensions", "Comfort", "describes comfort"),
    ("comfortably", "Size_Dimensions", "Comfort", "describes comfort"),
    
    # Special_User_Needs vs Target_User
    ("sensitive skin", "Target_User", "Special_User_Needs", "describes special user need"),
    ("cannot wear", "Target_User", "Special_User_Needs", "describes special user need"),
    ("eczema", "Target_User", "Special_User_Needs", "describes special user need"),
    ("allergy", "Target_User", "Special_User_Needs", "describes special user need"),
    ("neck sensitivity", "Target_User", "Special_User_Needs", "describes special user need"),
    ("skin sensitivity", "Target_User", "Special_User_Needs", "describes special user need"),
    
    # Functionality vs Usage_Scenario - more specific rules
    ("making greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("card making", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("greeting card", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("scrapbook", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("scrapbooking", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("decorating greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("decorates greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("create.*greeting card", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("creates.*greeting card", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("die-cut shapes for greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("shapes for greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("fastens.*greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("works on greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("peel and stick to cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("cards or scrapbook", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("flowers on greeting cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("for beginners", "Functionality", "Target_User", "describes target user"),
    
    # More Functionality -> Usage_Scenario
    ("card decoration", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("for card", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("paper crafting", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("create backgrounds for cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("card fronts", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("card backgrounds", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("card interior", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("graduation card", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("birthday cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("christmas cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("embellishing cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("used as christmas tree", "Functionality", "Special_Purpose", "describes special purpose"),
    
    # Functionality -> Performance
    ("retains sharpness", "Functionality", "Performance", "describes performance"),
    ("sharpness", "Functionality", "Performance", "describes performance"),
    ("creates intense colors", "Functionality", "Performance", "describes performance"),
    ("embosses beautifully", "Functionality", "Performance", "describes performance"),
    ("embosses beautifully", "Functionality", "Performance", "describes performance"),
    ("dries waterproof", "Functionality", "Performance", "describes performance"),
    ("glitter doesn't rub off", "Functionality", "Performance", "describes performance"),
    ("does not spread", "Functionality", "Performance", "describes performance"),
    
    # Functionality -> Product_Category
    ("die-cutting", "Functionality", "Product_Category", "is a product type"),
    ("die cutting", "Functionality", "Product_Category", "is a product type"),
    
    # Compatibility vs Brand_Preference
    ("works with", "Brand_Preference", "Compatibility", "describes compatibility"),
    ("compatible with", "Brand_Preference", "Compatibility", "describes compatibility"),
    ("fits", "Brand_Preference", "Compatibility", "describes compatibility"),
    
    # Performance vs Functionality
    ("works well", "Functionality", "Performance", "describes performance quality"),
    ("works great", "Functionality", "Performance", "describes performance quality"),
    ("performs well", "Functionality", "Performance", "describes performance quality"),
    ("cuts cleanly", "Functionality", "Performance", "describes performance"),
    ("cuts well", "Functionality", "Performance", "describes performance"),
    
    # Style_Design vs Appearance_Color
    ("pattern", "Appearance_Color", "Style_Design", "describes style/design"),
    ("design", "Appearance_Color", "Style_Design", "describes style/design"),
    
    # Ease_of_Use vs Compatibility
    ("easy to use with", "Ease_of_Use", "Compatibility", "describes compatibility"),
    ("works with", "Ease_of_Use", "Compatibility", "describes compatibility"),
    
    # Size_Dimensions vs Packaging_Quantity
    ("set of", "Size_Dimensions", "Packaging_Quantity", "describes packaging"),
    ("pack of", "Size_Dimensions", "Packaging_Quantity", "describes packaging"),
    ("piece set", "Size_Dimensions", "Packaging_Quantity", "describes packaging"),

    # Additional Functionality -> Usage_Scenario (usage scenarios vs functionality)
    ("can be used inside and outside cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("can be used for cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("usable in cards", "Functionality", "Usage_Scenario", "describes usage scenario"),
    ("doesn't add bulk to card", "Functionality", "Performance", "describes performance quality"),
    ("adds dimension to.*background", "Functionality", "Performance", "describes performance quality"),

    # Ease_of_Use -> Compatibility (machine compatibility)
    ("requires.*pass.*through machine", "Ease_of_Use", "Compatibility", "describes machine compatibility"),
    ("requires multiple passes", "Ease_of_Use", "Compatibility", "describes machine compatibility"),

    # Brand_Preference -> Compatibility (when describing requirements)
    ("requires.*sizzix", "Brand_Preference", "Compatibility", "describes compatibility requirement"),
    ("requires.*cottage cutz", "Brand_Preference", "Compatibility", "describes compatibility requirement"),
    ("requires.*spellbinders", "Brand_Preference", "Compatibility", "describes compatibility requirement"),

    # Compatibility -> Brand_Preference (when only product names)
    ("Big Kick by Sizzix", "Compatibility", "Brand_Preference", "is a product/brand name"),
    ("Spellbinders Wizard", "Compatibility", "Brand_Preference", "is a product/brand name"),
    ("Spellbinders Grand Calibur", "Compatibility", "Brand_Preference", "is a product/brand name"),

    # Quality_Craftsmanship -> Value (product variety)
    ("variety", "Quality_Craftsmanship", "Value", "describes product variety"),
    ("variety of", "Quality_Craftsmanship", "Value", "describes product variety"),
    ("amazing variety", "Quality_Craftsmanship", "Value", "describes product variety"),
    ("wide variety", "Quality_Craftsmanship", "Value", "describes product variety"),
    ("high quality variety", "Quality_Craftsmanship", "Value", "describes product variety"),
]

def apply_corrections(prefs: Dict) -> Dict:
    """Apply correction rules to preferences"""
    corrections_made = []

    # Check if pattern contains regex special characters
    def matches(entity: str, pattern: str) -> bool:
        entity_lower = entity.lower()
        pattern_lower = pattern.lower()
        if re.search(pattern_lower, entity_lower):
            return True
        return pattern_lower in entity_lower

    for category, category_data in prefs.items():
        if not isinstance(category_data, dict):
            continue

        # Collect all entities that need correction first (avoid modifying during iteration)
        entities_to_move = []  # List of (dimension, entity_dict, target_dim, reason)
        processed_entities = set()  # Track processed entities to avoid duplicates

        for dimension, entities in list(category_data.items()):
            if not isinstance(entities, list):
                continue

            for entity_dict in entities:
                if not isinstance(entity_dict, dict):
                    continue

                entity = entity_dict.get('entity', '')

                # Create a unique key for this entity to avoid duplicates
                entity_key = (category, dimension, entity.lower())

                # Skip if already processed
                if entity_key in processed_entities:
                    continue

                # Check each correction rule
                for pattern, source_dim, target_dim, reason in CORRECTION_RULES:
                    # Only apply rule if entity is in the expected source dimension
                    if dimension != source_dim:
                        continue

                    if matches(entity, pattern):
                        entities_to_move.append((dimension, entity_dict, target_dim, reason))
                        processed_entities.add(entity_key)
                        corrections_made.append({
                            'entity': entity_dict.get('entity'),
                            'from': f"{category}/{source_dim}",
                            'to': f"{category}/{target_dim}",
                            'reason': reason
                        })
                        break  # Only apply first matching rule

        # Now apply all corrections at once
        for source_dim, entity_dict, target_dim, reason in entities_to_move:
            # Remove from source dimension
            if source_dim in category_data:
                # Remove by identity (same object reference) or by value
                category_data[source_dim] = [
                    e for e in category_data[source_dim]
                    if e is not entity_dict and e.get('entity') != entity_dict.get('entity')
                ]

            # Add to target dimension
            if target_dim not in category_data:
                category_data[target_dim] = []

            # Mark the entity as corrected
            entity_dict = entity_dict.copy()  # Create a copy to avoid reference issues
            entity_dict['_corrected_from'] = source_dim
            entity_dict['_correction_reason'] = reason
            category_data[target_dim].append(entity_dict)

    return prefs, corrections_made


def clean_empty_dimensions(prefs: Dict) -> None:
    """Remove empty dimensions from preferences"""
    for category, category_data in prefs.items():
        if not isinstance(category_data, dict):
            continue
        for dimension in list(category_data.keys()):
            if not category_data[dimension] or len(category_data[dimension]) == 0:
                del category_data[dimension]


def deduplicate_corrections(corrections_list: List[Dict]) -> List[Dict]:
    """Remove duplicate correction entries"""
    seen = set()
    deduplicated = []

    for product_corr in corrections_list:
        unique_corrections = []
        seen_keys = set()

        for corr in product_corr.get('corrections', []):
            # Create a unique key for this correction
            key = (
                corr['entity'].lower().strip(),
                corr['from'],
                corr['to'],
                corr['reason']
            )

            if key not in seen_keys:
                seen_keys.add(key)
                unique_corrections.append(corr)

        if unique_corrections:
            deduplicated.append({
                'asin': product_corr['asin'],
                'type': product_corr['type'],
                'corrections': unique_corrections
            })

    return deduplicated


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1 Post-Processing: Fix Misclassified Preferences"
    )
    parser.add_argument(
        '--input', '-i',
        default="/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/preferences_A13OFOB1394G31.json",
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSON file path (default: overwrite input)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed correction information'
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output if args.output else input_file

    print("=" * 80)
    print("Stage 1 Post-Processing: Fix Misclassified Preferences")
    print("=" * 80)

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Input file: {input_file}")
    print(f"Loaded {data['total_products']} products")

    all_corrections = []
    corrections_count_before_dedup = 0

    # Process each product
    for i, product in enumerate(data['results']):
        asin = product.get('asin', f'unknown_{i}')

        # Process target user preferences
        target_prefs = product.get('target_user_preferences', {})
        if target_prefs:
            target_prefs, corrections = apply_corrections(target_prefs)
            if corrections:
                all_corrections.append({
                    'asin': asin,
                    'type': 'target',
                    'corrections': corrections
                })
                corrections_count_before_dedup += len(corrections)

        # Process other users preferences
        other_prefs = product.get('other_users_preferences', {})
        if other_prefs:
            other_prefs, corrections = apply_corrections(other_prefs)
            if corrections:
                all_corrections.append({
                    'asin': asin,
                    'type': 'other',
                    'corrections': corrections
                })
                corrections_count_before_dedup += len(corrections)

    # Deduplicate corrections
    print(f"\nDeduplicating corrections...")
    all_corrections = deduplicate_corrections(all_corrections)

    # Calculate statistics
    total_corrections = sum(len(item['corrections']) for item in all_corrections)
    duplicates_removed = corrections_count_before_dedup - total_corrections

    # Clean empty dimensions
    for product in data['results']:
        clean_empty_dimensions(product.get('target_user_preferences', {}))
        clean_empty_dimensions(product.get('other_users_preferences', {}))

    # Add metadata
    data['correction_timestamp'] = datetime.now().isoformat()
    data['total_corrections'] = total_corrections
    data['corrections'] = all_corrections

    # Save corrected data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total products processed: {data['total_products']}")
    print(f"Corrections before deduplication: {corrections_count_before_dedup}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final unique corrections: {total_corrections}")
    print(f"Products with corrections: {len(all_corrections)}")
    print(f"Output saved to: {output_file}")

    # Print sample corrections
    if all_corrections:
        print(f"\n{'='*80}")
        print("Sample Corrections (first 5 products)")
        print(f"{'='*80}")
        for item in all_corrections[:5]:
            print(f"\nProduct: {item['asin']} ({item['type']}) - {len(item['corrections'])} corrections")
            for corr in item['corrections'][:2]:
                print(f"  • {corr['entity'][:60]}")
                print(f"    {corr['from']} → {corr['to']}")
                print(f"    Reason: {corr['reason']}")

    # Verbose mode: print all corrections
    if args.verbose and all_corrections:
        print(f"\n{'='*80}")
        print("ALL CORRECTIONS (Verbose Mode)")
        print(f"{'='*80}")
        for item in all_corrections:
            print(f"\nProduct: {item['asin']} ({item['type']})")
            for corr in item['corrections']:
                print(f"  • {corr['entity']}")
                print(f"    {corr['from']} → {corr['to']}")
                print(f"    Reason: {corr['reason']}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
