#!/usr/bin/env python3
import json
import os
import random
import pickle
import csv
import asyncio
import sys

# Ensure stark/code is on Python path
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
# Also add user_perference for kb_helper
USER_PREF_DIR = os.path.join(CODE_DIR, "user_perference")
if USER_PREF_DIR not in sys.path:
    sys.path.append(USER_PREF_DIR)

try:
    from model import submit_batch_inference, wait_for_batch_results, set_api_responses_file
    from kb_helper import get_kb_instance
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class PreferenceEvolutionSampler:
    def __init__(self, user_pref_file):
        with open(user_pref_file, 'r') as f:
            data = json.load(f)
            self.product_data_list = data.get('products', [])
        
        self.kb = get_kb_instance()
        self.product_map = {p['asin']: p for p in self.product_data_list}
        
        # Load full metadata for titles
        META_PATH = '/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb/node_info.pkl'
        try:
            with open(META_PATH, 'rb') as f:
                self.full_meta = pickle.load(f)
            print(f"✅ Loaded full metadata for {len(self.full_meta)} products.")
        except Exception as e:
            self.full_meta = {}
            print(f"⚠️ Could not load full metadata: {e}")
        
        self.kb.load()
        
    def get_title(self, asin):
        # Full meta uses Node ID as keys
        node_id = self.kb.asin_to_id.get(asin)
        if node_id is None: return "Unknown Product"
        return self.full_meta.get(node_id, {}).get('title', 'Unknown Product')

    def _extract_seeds(self, product_data, asin):
        seeds = set()
        node_id = self.kb.asin_to_id.get(asin)
        if node_id is None: return seeds
        
        meta = self.full_meta.get(node_id, {})
        features = [f.lower() for f in meta.get('feature', [])]
        skb_attrs = self.kb.get_product_attributes(asin)
        meta_pool = features + [str(v).lower() for v_list in skb_attrs.values() for v in v_list]
        meta_pool += [k.lower() for k in skb_attrs.keys()]

        entities_groups = product_data.get('user_preference_entities', {})
        if isinstance(entities_groups, dict):
            for cat, items in entities_groups.items():
                if cat in ["Product Category", "Reasoning"]: continue
                if isinstance(items, list):
                    for item in items:
                        if item.get('sentiment') in ['positive', 'neutral']:
                            val = item.get('entity', '').lower().strip()
                            # Stricter definition: Must appear in metadata
                            if val and any(val in m for m in meta_pool):
                                seeds.add(val)
        return seeds

    def _extract_categories(self, product_data, asin):
        cats = set()
        node_id = self.kb.asin_to_id.get(asin)
        if node_id is None: return cats
        
        meta = self.full_meta.get(node_id, {})
        features = [f.lower() for f in meta.get('feature', [])]
        skb_attrs = self.kb.get_product_attributes(asin)
        meta_pool = [k.lower() for k in skb_attrs.keys()] + features + [str(v).lower() for v_list in skb_attrs.values() for v in v_list]

        entities_groups = product_data.get('user_preference_entities', {})
        if isinstance(entities_groups, dict):
            for cat, items in entities_groups.items():
                if cat in ["Product Category", "Reasoning"]: continue
                if isinstance(items, list):
                    for item in items:
                        if item.get('sentiment') in ['positive', 'neutral']:
                            val = item.get('entity', '').lower().strip()
                            # If the entity value itself is mentioned in metadata, we trust this category
                            if val and any(val in m for m in meta_pool):
                                cats.add(cat.strip().lower())
                                break
        return cats

    def _extract_all_entities_info(self, product_data, asin):
        all_ents = []
        node_id = self.kb.asin_to_id.get(asin)
        if node_id is None: return all_ents

        meta = self.full_meta.get(node_id, {})
        features = [f.lower() for f in meta.get('feature', [])]
        skb_attrs = self.kb.get_product_attributes(asin)
        meta_pool = [k.lower() for k in skb_attrs.keys()] + features + [str(v).lower() for v_list in skb_attrs.values() for v in v_list]

        entities_groups = product_data.get('user_preference_entities', {})
        if isinstance(entities_groups, dict):
            for cat, items in entities_groups.items():
                if cat in ["Product Category", "Reasoning"]: continue
                if isinstance(items, list):
                    for item in items:
                        val = item.get('entity', '').lower().strip()
                        cat_lower = cat.strip().lower()
                        # Strict Intersection: Entity text or Category name must exist in Product Metadata
                        if any(val in m for m in meta_pool) or any(cat_lower in m for m in meta_pool):
                            item_copy = item.copy()
                            item_copy['sort'] = "Category" if cat_lower == 'category' else cat
                            all_ents.append(item_copy)
        return all_ents

    async def verify_wishes_with_llm(self, candidate_pairs):
        to_verify = set()
        for pair in candidate_pairs:
            min_cat = pair.get('min_category', 'product')
            for wish_item in pair['pending_wishes']:
                to_verify.add((pair['target_asin'], wish_item['wish_text'], min_cat))
        
        if not to_verify: return {}

        verification_list = list(to_verify)
        prompts = []
        for asin_a, wish, min_cat in verification_list:
            skb_attrs = self.kb.get_product_attributes(asin_a)
            attr_str = json.dumps(skb_attrs, indent=1) if skb_attrs else "No specific attributes found."
            prod_a = self.product_map[asin_a]
            pos_ents = []
            for ent in self._extract_all_entities_info(prod_a, asin_a):
                if ent.get('sentiment') in ['positive', 'neutral']:
                    pos_ents.append(ent.get('entity', ''))
            pos_str = ", ".join(pos_ents) if pos_ents else "Not mentioned."
            prompt = f"Product Type: {min_cat}\nAttributes: {attr_str}\nPositive Features: {pos_str}\nWish: {wish}\nDoes this specific product satisfy this wish better than typical? YES/NO."
            prompts.append(prompt)

        print(f"🧠 Verifying {len(prompts)} wishes with LLM...")
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        results = await wait_for_batch_results(batch_id)
        verified_map = {}
        for i, res in enumerate(results):
            raw = res['response']['body']['choices'][0]['message']['content'].upper()
            verified_map[verification_list[i]] = "YES" in raw
        return verified_map

    async def verify_category_coverage_with_llm(self, pairs):
        """
        Verify if the attribute categories of A are semantically covered by B.
        """
        if not pairs: return {}

        prompts = []
        for pair in pairs:
            cats_a = ", ".join(sorted(list(pair['cats_a'])))
            cats_b = ", ".join(sorted(list(pair['cats_b'])))
            prompt = (
                f"Product A covers these attribute categories: [{cats_a}]\n"
                f"Product B covers these attribute categories: [{cats_b}]\n"
                "Semantically, does B cover at least two core attribute dimensions of A? "
                "Respond ONLY YES or NO."
            )
            prompts.append(prompt)

        print(f"🧠 Semantically verifying category coverage for {len(prompts)} pairs...")
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        results = await wait_for_batch_results(batch_id)
        
        coverage_map = {}
        for i, res in enumerate(results):
            raw = res['response']['body']['choices'][0]['message']['content'].upper()
            coverage_map[i] = "YES" in raw
        return coverage_map

    async def verify_feature_relevance_with_llm(self, pairs):
        """
        Verify if a product feature semantically addresses a user preference (wish/pain).
        pairs: list of (feature_text, preference_text)
        """
        if not pairs: return []
        
        # Deduplicate to save tokens
        unique_pairs = list(set(pairs))
        pair_to_idx = {p: i for i, p in enumerate(unique_pairs)}
        
        prompts = []
        for feat, pref in unique_pairs:
            prompt = (
                f"User Preference: \"{pref}\"\n"
                f"Product feature: \"{feat}\"\n"
                "Could this product feature potentially help with or be relevant to the user preference? "
                "Respond ONLY YES or NO."
            )
            prompts.append(prompt)
            
        print(f"🧠 Verifying {len(prompts)} feature-preference pairs with LLM...")
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        results = await wait_for_batch_results(batch_id)
        
        results_map = {}
        for i, res in enumerate(results):
            raw = res['response']['body']['choices'][0]['message']['content'].upper()
            results_map[unique_pairs[i]] = "YES" in raw
            
        # Map back to original list order
        return [results_map.get(p, False) for p in pairs]

    async def verify_entity_in_metadata_with_llm(self, pairs):
        """
        Verify if a user preference entity is semantically mentioned in product metadata.
        pairs: list of (entity_text, original_text, metadata_combined_text)
        Returns: list of bool (True if entity is mentioned in metadata)
        """
        if not pairs: return []
        
        # Deduplicate
        unique_pairs = list(set(pairs))
        
        prompts = []
        for entity, original, metadata in unique_pairs:
            prompt = (
                f"User Attribute: \"{entity}\"\n"
                f"User's Original Context: \"{original}\"\n"
                f"Product Metadata: \"{metadata}\"\n"
                "Based on the product metadata, does this product possess or support the specific attribute as described in the user's context? "
                "Respond ONLY YES or NO."
            )
            prompts.append(prompt)
            
        print(f"🧠 Verifying {len(prompts)} entity-metadata pairs with LLM...")
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        results = await wait_for_batch_results(batch_id)
        
        results_map = {}
        for i, res in enumerate(results):
            raw = res['response']['body']['choices'][0]['message']['content'].upper()
            results_map[unique_pairs[i]] = "YES" in raw
            
        # Map back to original list order
        return [results_map.get(p, False) for p in pairs]

    async def find_evolution_pairs(self):
        valid_asins = list(self.product_map.keys())
        print(f"🔍 Analyzing {len(valid_asins)} products for evolution pairs...")
        
        # Step 1: Collect all entity-metadata pairs for LLM verification
        all_entity_metadata_pairs = []
        asin_to_entities = {}  # Store entities per ASIN for later lookup
        
        print("📦 Collecting entity-metadata pairs for verification...")
        for asin in valid_asins:
            prod = self.product_map[asin]
            if not prod.get('user_preference_entities'):
                continue
                
            node_id = self.kb.asin_to_id.get(asin)
            if node_id is None:
                continue
                
            # Build metadata text
            meta = self.full_meta.get(node_id, {})
            title = meta.get('title', '')
            description_raw = meta.get('description', [])
            description = " ".join(description_raw) if isinstance(description_raw, list) else str(description_raw)
            features = meta.get('feature', [])
            price = meta.get('price', '')
            details = meta.get('details', {})
            skb_attrs = self.kb.get_product_attributes(asin)
            
            # Combine all metadata into one text block
            parts = []
            if title: parts.append(f"Title: {title}")
            if price: parts.append(f"Price: {price}")
            if description: parts.append(f"Description: {description}")
            if features: parts.append("Features: " + " | ".join(features))
            if details:
                details_str = ", ".join([f"{k}: {v}" for k, v in details.items() if v])
                if details_str: parts.append(f"Details: {details_str}")
            
            for k, vals in skb_attrs.items():
                if k.lower() not in ['asin', 'brand']:
                    parts.append(f"{k}: {', '.join(str(v) for v in vals)}")
            
            metadata_text = " | ".join(parts)
            
            # Collect all entities from this product
            entities_groups = prod.get('user_preference_entities', {})
            product_entities = []
            
            if isinstance(entities_groups, dict):
                for cat, items in entities_groups.items():
                    if cat in ["Product Category", "Reasoning"]:
                        continue
                    if isinstance(items, list):
                        for item in items:
                            if item.get('sentiment') in ['positive', 'neutral']:
                                val = item.get('entity', '').strip()
                                original = item.get('original_text', val).strip()
                                if val:
                                    product_entities.append({
                                        'category': cat,
                                        'entity': val,
                                        'original': original,
                                        'item': item
                                    })
                                    all_entity_metadata_pairs.append((val, original, metadata_text))
            
            asin_to_entities[asin] = {
                'entities': product_entities,
                'metadata_text': metadata_text
            }
        
        # Step 2: Batch verify all pairs with LLM
        print(f"🚀 Batch verifying {len(all_entity_metadata_pairs)} entity-metadata pairs...")
        llm_verification_results = await self.verify_entity_in_metadata_with_llm(all_entity_metadata_pairs)
        
        # Step 3: Build lookup dict
        verification_lookup = {}
        for i, pair in enumerate(all_entity_metadata_pairs):
            verification_lookup[pair] = llm_verification_results[i]
        
        # Step 4: Extract valid categories and seeds using LLM results
        asin_to_valid_cats = {}
        asin_to_valid_seeds = {}
        
        for asin, data in asin_to_entities.items():
            valid_cats = set()
            valid_seeds = set()
            metadata_text = data['metadata_text']
            
            for ent_data in data['entities']:
                entity = ent_data['entity']
                original = ent_data['original']
                category = ent_data['category']
                
                # Check if LLM verified this entity
                if verification_lookup.get((entity, original, metadata_text), False):
                    valid_cats.add(category.strip().lower())
                    valid_seeds.add(entity.lower())
            
            asin_to_valid_cats[asin] = valid_cats
            asin_to_valid_seeds[asin] = valid_seeds
        
        # Step 5: Initial aggregation for ALL valid ASINs with enough seeds
        aggregated_results = {}
        for asin_a in valid_asins:
            seeds_a = asin_to_valid_seeds.get(asin_a, set())
            if len(seeds_a) < 3: # 🔴 STANDALONE THRESHOLD: Must have at least 3 verified preferences
                continue
                
            prod_a = self.product_map[asin_a]
            seed_map = {e['entity'].lower().strip(): e for e in asin_to_entities[asin_a]['entities'] if e['item'].get('sentiment') in ['positive', 'neutral']}
            
            aggregated_results[asin_a] = {
                'target_asin': asin_a,
                'neighbors': [],
                'min_category': prod_a.get('min_category'),
                'full_augmented_pool': [],
                'seeds_a': seeds_a,
                'seed_map': seed_map,
                'sorts_covered_by_a': set()
            }
            
            # Initialize pool with target's own seeds
            for seed_text in seeds_a:
                if seed_text in seed_map:
                    orig = seed_map[seed_text]
                    s_val = orig.get('category', 'Unknown')
                    aggregated_results[asin_a]['sorts_covered_by_a'].add(s_val.lower().strip())
                    aggregated_results[asin_a]['full_augmented_pool'].append({
                        "entity": seed_text, "original_text": orig.get('entity'),
                        "sort": s_val, "sentiment": orig.get('sentiment'),
                        "source_product": asin_a, "source_type": "target_base_seed"
                    })

        # Step 6: Build candidate pairs using LLM-verified categories (Optional Enhancement)
        candidate_pairs = []
        for asin_a in aggregated_results.keys():
            prod_a = self.product_map[asin_a]
            cats_a = asin_to_valid_cats.get(asin_a, set())
            
            for asin_b in valid_asins:
                if asin_a == asin_b: continue
                prod_b = self.product_map[asin_b]
                if prod_a.get('min_category') != prod_b.get('min_category'): continue
                
                cats_b = asin_to_valid_cats.get(asin_b, set())
                if not cats_b: continue
                
                candidate_pairs.append({
                    'target_asin': asin_a, 'neighbor_asin': asin_b,
                    'cats_a': cats_a, 'cats_b': cats_b,
                    'prod_a': prod_a, 'prod_b': prod_b
                })
        
        if candidate_pairs:
            print(f"📊 Found {len(candidate_pairs)} potential neighbor pairs. Verifying coverage...")
            # 1. Semantic Category Coverage Verification
            coverage_results = await self.verify_category_coverage_with_llm(candidate_pairs)
            
            # 2. Refine candidates based on coverage
            refined_candidates = []
            for i, cand in enumerate(candidate_pairs):
                if coverage_results.get(i):
                    asin_a, asin_b = cand['target_asin'], cand['neighbor_asin']
                    prod_a, prod_b = cand['prod_a'], cand['prod_b']
                    
                    seeds_a = asin_to_valid_seeds.get(asin_a, set())
                    ents_b = self._extract_all_entities_info(prod_b, asin_b)
                    pending_wishes, regular_extras = [], []
                    for ent in ents_b:
                        val = ent.get('entity', '').lower().strip()
                        if val in seeds_a: continue
                        sentiment = ent.get('sentiment')
                        if sentiment in ['positive', 'neutral']:
                            regular_extras.append(ent)
                        elif sentiment == 'negative':
                            wish = ent.get('improvement_wish')
                            if wish and wish.strip():
                                pending_wishes.append({"wish_text": wish.strip(), "original_item": ent})
                    
                    if pending_wishes or regular_extras:
                        refined_candidates.append({
                            'target_asin': asin_a, 'neighbor_asin': asin_b,
                            'pending_wishes': pending_wishes, 'regular_extras': regular_extras,
                        })

            if refined_candidates:
                # 3. LLM Wish Verification
                wish_results = await self.verify_wishes_with_llm(refined_candidates)
                
                # Update aggregated_results with neighbor info
                for cand in refined_candidates:
                    asin_a, asin_b = cand['target_asin'], cand['neighbor_asin']
                    target_entry = aggregated_results[asin_a]
                    neighbor_pool = []
                    neg_resolved_count = 0
                    
                    for ent in cand['regular_extras']:
                        sort_val = ent.get('sort', 'Unknown').lower().strip()
                        if sort_val in target_entry['sorts_covered_by_a']: continue
                        neighbor_pool.append({
                            "entity": ent.get('entity').lower().strip(), "original_text": ent.get('entity'),
                            "sort": ent.get('sort', 'Unknown'), "sentiment": ent.get('sentiment'),
                            "source_product": asin_b, "source_type": "neighbor_extra"
                        })
                    
                    for wish_item in cand['pending_wishes']:
                        wish_text = wish_item['wish_text']
                        if wish_results.get((asin_a, wish_text)):
                            ent = wish_item['original_item']
                            wish_sort = ent.get('sort', 'Unknown').lower().strip()
                            if wish_sort in target_entry['sorts_covered_by_a']: continue
                            neighbor_pool.append({
                                "entity": wish_text, "original_text": ent.get('entity'),
                                "sort": ent.get('sort', 'Unknown'), "sentiment": "negative_resolved",
                                "source_product": asin_b, "source_type": "neighbor_wish_transfer"
                            })
                            neg_resolved_count += 1
                    
                    if neighbor_pool:
                        target_entry['neighbors'].append({'asin': asin_b, 'neg_resolved': neg_resolved_count})
                        target_entry['full_augmented_pool'].extend(neighbor_pool)
                        for p in neighbor_pool:
                            target_entry['sorts_covered_by_a'].add(p['sort'].lower().strip())

        # Return all query candidates (even those with no neighbors)
        return list(aggregated_results.values())

async def main():
    USER_PREF_FILE = '/home/wlia0047/ar57/wenyu/result/user_preference_entities.json'
    OUTPUT_FILE = '/home/wlia0047/ar57/wenyu/result/generated_kg_queries.json'
    CSV_FILE = '/home/wlia0047/ar57/wenyu/result/query/kg_queries_clean.csv'
    API_LOG_FILE = '/home/wlia0047/ar57/wenyu/result/generate_queries_api_raw.json'
    sampler = PreferenceEvolutionSampler(USER_PREF_FILE)
    evolution_pairs = await sampler.find_evolution_pairs()
    if not evolution_pairs: return
    
    prompts, metadata_map = [], {}
    set_api_responses_file(API_LOG_FILE, overwrite=True)

    # 1. Collect all Feature-Preference pairs for batch verification
    all_verification_pairs = []
    # Store mapping to retrieve pairs later: idx -> list of (feat, pref)
    pair_verification_map = {}

    print("🔍 collecting pairs for semantic verification...")
    
    for idx, pair in enumerate(evolution_pairs):
        asin_a = pair['target_asin']
        node_id_a = sampler.kb.asin_to_id.get(asin_a)
        meta_a = sampler.full_meta.get(node_id_a, {})
        raw_features_a = meta_a.get('feature', [])
        
        # Aggregate Pains/Wishes
        # Aggregate Pains/Wishes from neighbors (Optional)
        all_neighbor_asins = [n['asin'] for n in pair['neighbors']]
        all_b_pains, all_b_wishes = [], []
        for nb_asin in all_neighbor_asins:
            nb_data = sampler.product_map.get(nb_asin, {})
            nb_prefs = nb_data.get('user_preference_entities', {})
            for cat_prefs in nb_prefs.values():
                for p in cat_prefs:
                    if p.get('sentiment') == 'negative':
                        all_b_pains.append(p.get('entity', '').lower())
                        if p.get('improvement_wish'):
                            all_b_wishes.append(p.get('improvement_wish', '').lower())
        
        unique_pains = list(set(all_b_pains))[:5]
        unique_wishes = list(set(all_b_wishes))[:5]
        pair['unique_pains'] = unique_pains
        pair['unique_wishes'] = unique_wishes
        
    # Skipped Phase 4: Feature-Preference Verification per user request.
    # We now trust Phase 1 (Entity-Metadata) implies the product has these qualities.
    
    # 3. Construct Prompts using Verified User Preferences directly
    for idx, pair in enumerate(evolution_pairs):
        min_cat = pair['min_category']
        asin_a = pair['target_asin']
        pool = pair['full_augmented_pool']
        
        # We use the full pool which includes A's seeds AND neighbor's converted wishes/extras
        # User requested max 3 attributes
        all_seeds = [obj['origin_text'] if 'origin_text' in obj else obj.get('original_text', obj['entity']) 
                       for obj in pool]
        # In find_evolution_pairs, the pool is constructed: seeds_a + neighbor_pool
        # We take the top 3.
        essence_values = all_seeds[:3]
        
        skb_attrs = sampler.kb.get_product_attributes(asin_a)
        brand_a = skb_attrs.get('Brand', [''])[0]

        if not essence_values:
            print(f"⚠️ Skipping ASIN {asin_a}: No verified user preferences found (Unexpected). Pool size: {len(pool)}")
            continue

        # Prepare detailed attributes for logging
        detailed_attributes = []
        
        # Conditional Brand Injection
        brand_mentioned_in_pref = False
        if brand_a:
            target_pref_entities = [obj['entity'] for obj in pool if obj['source_product'] == asin_a]
            for entity_text in target_pref_entities:
                if brand_a.lower() in entity_text.lower():
                    brand_mentioned_in_pref = True
                    break
        
        brand_segment = ""
        if brand_mentioned_in_pref:
            detailed_attributes.append({"value": brand_a, "type": "Brand", "source": asin_a, "sentiment": "positive"})
            brand_segment = f' from brand "{brand_a}"'

        # Log utilized preferences
        for val in essence_values:
            detailed_attributes.append({
                "value": val,
                "type": "User Preference / Converted Wish",
                "source": "Combined Pool",
                "sentiment": "positive"
            })
            
        # 3. Simple Human Prompt
        prompt = f"""You are a human shopper searching for a specific {min_cat} on Amazon.
You are looking for a {min_cat} that has the following key attributes/features:
{chr(10).join([f"- {s}" for s in essence_values])}

YOUR TASK:
Write a concise, natural search query (25-30 words) to find such a product using FIRST-PERSON perspective (e.g., "I need...", "I am looking for...").
The tone should be natural, direct, and sound like a real human typing into a search bar describing their specific needs.
Do NOT roleplay as a specific character (like "expert" or "frustrated user"). Just be a normal user expressing their own intent.


OUTPUT FORMAT:
Output ONLY the query text. No quotes."""
        pair['detailed_attributes_used'] = detailed_attributes
        prompt_idx = len(prompts)
        prompts.append(prompt); metadata_map[prompt_idx] = pair
    
    print(f"🚀 Submitting batch for {len(prompts)} queries...")
    batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
    results = await wait_for_batch_results(batch_id)
    final_queries = []
    for res in results:
        custom_id = res.get('custom_id')
        if custom_id and custom_id.startswith('req-'):
            try:
                idx = int(custom_id.split('-')[1])
                pair = metadata_map.get(idx)
                if pair:
                    # SiliconFlow batch result format has the response in res['response']
                    response_obj = res.get('response', {})
                    body = response_obj.get('body', {})
                    choices = body.get('choices', [])
                    if choices:
                        content = choices[0].get('message', {}).get('content', '')
                        import re; content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip().strip('"')
                        content = re.sub(r'^(Query|Shopping Query|Search:|Keywords:)\s*', '', content, flags=re.IGNORECASE).strip()
                        final_queries.append({
                            "id": idx, 
                            "query": content.strip('*').strip('"'), 
                            "target_asin": pair['target_asin'], 
                            "neighbor_asins": [n['asin'] for n in pair['neighbors']],
                            "attributes_used": pair['detailed_attributes_used'], 
                            "common_category": pair['min_category']
                        })
                    else:
                        print(f"No choices found in response for {custom_id}")
                else:
                    print(f"No metadata map entry found for index {idx}")
            except Exception as e: print(f"Error processing {custom_id}: {e}")
        else:
            print(f"Skipping result with unexpected custom_id: {custom_id}")
                
    with open(OUTPUT_FILE, 'w') as f: json.dump(final_queries, f, indent=2)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['id', 'clean_query', 'answer_ids'])
        for item in final_queries:
            # Convert ASIN to SKB Node ID
            node_id = sampler.kb.asin_to_id.get(item['target_asin'])
            if node_id is None:
                print(f"⚠️ Warning: No Node ID found for ASIN {item['target_asin']}")
                node_id = -1 # Or skip
            writer.writerow([item['id'], item['query'], json.dumps([node_id])])
    print(f"✅ Saved {len(final_queries)} queries.")

if __name__ == "__main__":
    asyncio.run(main())