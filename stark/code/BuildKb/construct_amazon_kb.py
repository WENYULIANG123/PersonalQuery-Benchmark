
import json
import os
import pickle
import torch
from collections import defaultdict
from tqdm import tqdm

# Define paths for raw data
RAW_DATA_DIR = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018"
META_FILE = "meta_Arts_Crafts_and_Sewing.json"
REVIEW_FILE = "Arts_Crafts_and_Sewing.json"

def load_jsonl(file_path):
    """Helper to load JSONL file"""
    data = []
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def construct_kb(kg_data_dir, output_dir):
    """
    Constructs the STARK Knowledge Base from normalized JSON files.
    Enriches product nodes with raw metadata and reviews.
    
    Files expected in kg_data_dir:
    - direct_attributes_collection.json (brand, category, price)
    - style_values_collection.json (color, size, style, material, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Attribute Data
    print("Loading normalized attribute files...")
    with open(os.path.join(kg_data_dir, "direct_attributes_collection.json"), 'r') as f:
        direct_data = json.load(f)
    with open(os.path.join(kg_data_dir, "style_values_collection.json"), 'r') as f:
        style_data = json.load(f)

    # 1.1 Load Raw Metadata and Reviews for enrichment
    print("Loading raw metadata and reviews for enrichment...")
    meta_path = os.path.join(RAW_DATA_DIR, META_FILE)
    reviews_path = os.path.join(RAW_DATA_DIR, REVIEW_FILE)
    
    # Load metadata into a dict: ASIN -> Metadata Dict
    raw_meta_list = load_jsonl(meta_path)
    asin_to_meta = {item['asin']: item for item in raw_meta_list if 'asin' in item}
    print(f"Loaded metadata for {len(asin_to_meta)} products.")
    del raw_meta_list # Free memory
    
    # Load reviews into a dict: ASIN -> List of Reviews
    raw_reviews_list = load_jsonl(reviews_path)
    asin_to_reviews = defaultdict(list)
    for review in raw_reviews_list:
        if 'asin' in review:
            # Standardize review format to avoid KeyErrors during doc_info generation
            cleaned_review = {
                'reviewerID': review.get('reviewerID', ''),
                'summary': review.get('summary', ''),
                'reviewText': review.get('reviewText', ''),
                'overall': review.get('overall', None),
                'vote': review.get('vote', '0'),
                'verified': review.get('verified', False),
                'reviewTime': review.get('reviewTime', ''),
                'asin': review.get('asin', '')
            }
            asin_to_reviews[review['asin']].append(cleaned_review)
    print(f"Loaded reviews for {len(asin_to_reviews)} products.")
    del raw_reviews_list # Free memory

    # 2. Build Attribute Maps
    asin_to_attrs = defaultdict(dict)
    
    # Process Direct Attributes
    for attr, values in direct_data.items():
        for val, asins in values.items():
            if not val: continue
            for asin in asins:
                if attr not in asin_to_attrs[asin]:
                    asin_to_attrs[asin][attr] = []
                asin_to_attrs[asin][attr].append(val)
                
    # Process Style/Review Attributes
    for attr, values in style_data.items():
        for val, asins in values.items():
            if not val: continue
            for asin in asins:
                if attr not in asin_to_attrs[asin]:
                    asin_to_attrs[asin][attr] = []
                asin_to_attrs[asin][attr].append(val)

    # 3. Assign IDs
    node_info = {}
    next_id = 0
    
    asin_to_node_id = {}
    attr_val_to_node_id = {} # (attr_type, val) -> id
    
    print("Creating Nodes...")
    # Products first
    # Use the union of ASINs from attributes and metadata to ensure coverage? 
    # Or strict adherence to 'asin_to_attrs' (only products with extracted attributes)?
    # Usually we want nodes in the graph, so we stick to 'asin_to_attrs'.
    
    for asin in tqdm(list(asin_to_attrs.keys()), desc="Product Nodes"):
        nid = next_id
        next_id += 1
        asin_to_node_id[asin] = nid
        
        # Base node info
        node_data = {
            'node_type': 'product',
            'node_key': asin,
            'lookup_key': asin,
            'asin': asin
        }
        
        # Enrich with metadata
        if asin in asin_to_meta:
            meta = asin_to_meta[asin]
            # Add requested fields matching the 'big file' structure
            node_data['title'] = meta.get('title', '')
            node_data['description'] = meta.get('description', '')
            node_data['feature'] = meta.get('feature', [])
            node_data['price'] = meta.get('price', '')
            node_data['details'] = meta.get('details', {})
            node_data['category'] = meta.get('category', []) # Raw category path
            # 'brand' might be in meta as well
            if 'brand' in meta:
                node_data['brand'] = meta['brand']
        
        # Enrich with reviews
        if asin in asin_to_reviews:
            node_data['review'] = asin_to_reviews[asin]
        else:
            node_data['review'] = []
            
        node_info[nid] = node_data
        
    # Attribute Nodes
    for asin, attrs in tqdm(asin_to_attrs.items(), desc="Attribute Nodes"):
        for attr_type, vals in attrs.items():
            for val in vals:
                key = (attr_type, val)
                if key not in attr_val_to_node_id:
                    nid = next_id
                    next_id += 1
                    attr_val_to_node_id[key] = nid
                    node_info[nid] = {
                        'node_type': attr_type.lower().replace(" ", "_"),
                        'node_key': val,
                        'lookup_key': f"{attr_type.lower().replace(' ', '_')}::{val}",
                        'value': val,
                        'attribute_type': attr_type
                    }

    # 4. Build Edges
    edges = []
    edge_type_dict = {}
    attr_type_to_edge_id = {}
    next_edge_type_id = 0
    
    print("Building Edges...")
    for asin, attrs in tqdm(asin_to_attrs.items(), desc="Edges"):
        if asin not in asin_to_node_id: continue # Should not happen based on loop above
        src_id = asin_to_node_id[asin]
        
        for attr_type, vals in attrs.items():
            # Standardize edge names: e.g. "Color" -> "has_color"
            edge_name = f"has_{attr_type.lower().replace(' ', '_')}"
            if edge_name not in attr_type_to_edge_id:
                eid = next_edge_type_id
                next_edge_type_id += 1
                attr_type_to_edge_id[edge_name] = eid
                edge_type_dict[eid] = edge_name
            
            edge_type_id = attr_type_to_edge_id[edge_name]
            
            for val in vals:
                if (attr_type, val) in attr_val_to_node_id:
                    dst_id = attr_val_to_node_id[(attr_type, val)]
                    edges.append((src_id, dst_id, edge_type_id))

    # 5. Save Files
    print(f"Saving Knowledge Base to {output_dir}...")
    
    # asin_mapping.pkl
    with open(os.path.join(output_dir, 'asin_mapping.pkl'), 'wb') as f:
        pickle.dump({'asin_to_node_id': asin_to_node_id}, f)
        
    # node_info.pkl
    with open(os.path.join(output_dir, 'node_info.pkl'), 'wb') as f:
        pickle.dump(node_info, f)
        
    # edge_type_dict.pkl
    with open(os.path.join(output_dir, 'edge_type_dict.pkl'), 'wb') as f:
        pickle.dump(edge_type_dict, f)
        
    # edge_index.pt and edge_types.pt (Tensors)
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_types = torch.tensor([e[2] for e in edges], dtype=torch.long)
    
    torch.save(edge_index, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(edge_types, os.path.join(output_dir, 'edge_types.pt'))
    
    print("Done! KB Construction Complete.")
    print(f"Total Nodes: {len(node_info)}")
    print(f"Total Edges: {len(edges)}")

if __name__ == "__main__":
    KG_DATA_DIR = "/home/wlia0047/ar57/wenyu/result/KgData"
    OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb"
    construct_kb(KG_DATA_DIR, OUTPUT_DIR)
