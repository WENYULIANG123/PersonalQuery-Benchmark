
import json
import re

FILE_PATH = "/home/wlia0047/ar57/wenyu/result/entity_matching_results_normalized.json"

def fix_json_structure():
    print(f"Reading {FILE_PATH}...")
    with open(FILE_PATH, 'r') as f:
        content = f.read()

    # Fix the double brace issue introduced by sed
    # Pattern: Look for { followed immediately by { with whitespace
    # It seems the sed left something like:
    # {
    # {
    # Or just an empty { inside a list
    
    # Let's try to remove lines that are just "          {" if the previous line was also "          {"
    # Actually, simpler: load, if fail, try to patch text.
    
    # Specific patch for the "quality" deletion error
    # We saw in view_file:
    # 1509:           {
    # 1510:           {
    # 1511:             "entity": "easy cleanup",
    
    # We want to remove one of those {
    
    # We also need to handle the case where "quality" was the last item and we removed }, leaving {
    
    # 1. Regex replace double opening braces
    # This matches cases where we have two opening braces on separate lines with same indentation
    fixed_content = re.sub(r'(\s*\{\n)\1', r'\1', content)
    
    # 2. Also check for cases where we might have empty objects {} left or similar
    # But first let's see if that fixes the load
    
    try:
        data = json.loads(fixed_content)
        print("JSON structure repaired correctly.")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON repair failed: {e}")
        # Fallback: Manual line processing if needed, but regex should catch the dupes
        # Let's try to be more aggressive with the patch if needed
        # maybe there's a trailing comma issue too?
        # If we deleted the last item in a list, the previous item has a comma.
        # But we deleted specific items, so comma handling is tricky with regex.
        # However, our sed was: delete line match + 2 lines.
        # Format: { "entity": "quality", "sentiment": "pos" },
        # If we deleted 3 lines, we kept the trailing comma if it was on the closing brace line?
        # Wait, usually closing brace is on new line `},`
        # If we successfully deleted `},` then we might be ok comma-wise for the previous item?
        # NO, the previous item would still have a comma.
        # e.g.
        # { A },
        # { B },  <-- deleted
        # { C }
        # becomes:
        # { A },
        # { C }
        # This is valid.
        
        # But what if we deleted:
        # {
        #   "entity": "quality",
        #   "sentiment": "pos"
        # }
        # We start at "entity", delete 2 more. That deletes "sentiment" and "}".
        # So we leave the opening {.
        # That explains the double {{ we saw.
        
        # So yes, removing the extra { should fix it.
        # But wait, if we left an opening {, does the next item have an opening {?
        # Original:
        # {  <-- Line A
        #   "entity": "quality", <-- Line B (Match)
        #   "sentiment": "pos"   <-- Line C
        # }                      <-- Line D
        #
        # Sed `/,+2d` on Line B deletes B, C, D.
        # We are left with Line A ({).
        # And then the next item starts with {.
        # So we have:
        # {
        # {
        #   "entity": "next"...
        #
        # So yes, removing one { is correct.
        
        # But what about commas?
        # Line A did not have a comma.
        # The previous item (before quality) had `},`.
        # So:
        # },
        # {  (Empty one left from quality)
        # {  (Start of next item)
        #
        # If we remove one {, we have:
        # },
        # {
        #   "entity": "next"
        #
        # This is valid!
        
        # What if "quality" was the LAST item?
        # },
        # { (Leftover)
        # ]
        # That would be invalid: `... }, { ]`
        # We'd have a trailing comma on previous item, and an empty object start?
        # We need to handle `{\s*]` pattern -> `]`
        
        fixed_content = re.sub(r'\{\s*\]', ']', fixed_content)
        
        try:
             data = json.loads(fixed_content)
             return data
        except Exception as e2:
             print(f"Still failed: {e2}")
             return None

def clean_data(data):
    if not data: return
    
    count_size = 0
    count_quality = 0
    count_beginner = 0
    
    def clean_entity_list(ent_list):
        nonlocal count_size, count_quality, count_beginner
        files_to_keep = []
        if not isinstance(ent_list, list): return ent_list
        
        new_list = []
        for item in ent_list:
            if not isinstance(item, dict):
                new_list.append(item)
                continue
                
            entity_text = item.get("entity", "").strip()
            
            # 1. Remove "size" dimensions/entities
            # Check case-insensitive
            if entity_text.lower() == "size":
                count_size += 1
                continue
                
            # 2. Remove "quality"
            if entity_text.lower() == "quality":
                count_quality += 1
                continue
                
            # 3. Remove "Beginner"
            if entity_text.lower() == "beginner":
                count_beginner += 1
                continue
            
            new_list.append(item)
        return new_list

    products = data.get("products", [])
    for p in products:
        user_prefs = p.get("user_preference_entities", {})
        for cat, entities in user_prefs.items():
            user_prefs[cat] = clean_entity_list(entities)
            
    print(f"Removed {count_size} 'size' entities.")
    print(f"Removed {count_quality} 'quality' entities.")
    print(f"Removed {count_beginner} 'beginner' entities.")
    
    return data

def main():
    data = fix_json_structure()
    if data:
        data = clean_data(data)
        
        print(f"Saving to {FILE_PATH}...")
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Done.")
    else:
        print("Could not fix JSON structure.")

if __name__ == "__main__":
    main()
