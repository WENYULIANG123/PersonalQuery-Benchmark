import os
import pickle
import torch
import json
from collections import defaultdict

class AttributeKnowledgeBase:
    def __init__(self, kb_dir):
        self.kb_dir = kb_dir
        self.node_info = None
        self.edge_index = None
        self.edge_types = None
        self.asin_mapping = None
        self.edge_type_dict = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        print(f"Loading Knowledge Base from {self.kb_dir}...")
        
        # Load ASIN mapping
        with open(os.path.join(self.kb_dir, 'asin_mapping.pkl'), 'rb') as f:
            mapping_data = pickle.load(f)
            self.asin_to_id = mapping_data['asin_to_node_id']
        
        # Load Node Info
        with open(os.path.join(self.kb_dir, 'node_info.pkl'), 'rb') as f:
            self.node_info = pickle.load(f)
            
        # Load Edge Type Dict
        with open(os.path.join(self.kb_dir, 'edge_type_dict.pkl'), 'rb') as f:
            self.edge_type_dict = pickle.load(f)
            
        # Load Edges
        self.edge_index = torch.load(os.path.join(self.kb_dir, 'edge_index.pt'))
        self.edge_types_tensor = torch.load(os.path.join(self.kb_dir, 'edge_types.pt'))
        
        self._loaded = True
        print("Knowledge Base loaded successfully.")

    def get_product_attributes(self, asin):
        if not self._loaded:
            self.load()
            
        if asin not in self.asin_to_id:
            return {}
            
        node_id = self.asin_to_id[asin]
        
        # Find edges where src == node_id
        # edge_index is (2, num_edges), row 0 is src, row 1 is dst
        mask = self.edge_index[0] == node_id
        neighbor_indices = mask.nonzero().flatten()
        
        if len(neighbor_indices) == 0:
            return {}
            
        attributes = defaultdict(list)
        
        for idx in neighbor_indices:
            dst_id = self.edge_index[1][idx].item()
            edge_type_id = self.edge_types_tensor[idx].item()
            
            edge_name = self.edge_type_dict.get(edge_type_id, 'unknown')
            
            # Use edge name as category (e.g. 'has_color' -> 'Color')
            category = edge_name.replace('has_', '').replace('_', ' ').title()
            
            node_data = self.node_info.get(dst_id)
            if node_data:
                value = node_data.get('node_key') # The attribute value
                if value:
                    attributes[category].append(value)
                    
                    
        return dict(attributes)

    def get_product_unstructured_info(self, asin):
        """
        Get unstructured information (title, description, feature, price, details) for a product.
        """
        if not self._loaded:
            self.load()
            
        if asin not in self.asin_to_id:
            return {}
            
        node_id = self.asin_to_id[asin]
        node_data = self.node_info.get(node_id, {})
        
        info = {
            'title': node_data.get('title', ''),
            'description': node_data.get('description', ''),
            'feature': node_data.get('feature', []),
            'price': node_data.get('price', ''),
            'details': node_data.get('details', {})
        }
        return info

    def get_min_category(self, asin):
        """
        Get the most specific category (leaf category) for a product.
        Logic:
        1. Scan 'Category' and 'Main Category' attributes.
        2. If 'Category' exists, pick the most specific one (heuristic: longest string).
        3. Else fallback to 'Main Category'.
        """
        attrs = self.get_product_attributes(asin)
        
        best_category = None
        
        # 1. Try consolidated 'Category' attribute first
        if 'Category' in attrs:
            candidates = [v for v in attrs['Category'] if v]
            if candidates:
                # Heuristic: The most specific category is usually the longest string
                # or the one that appears last in the original Amazon breadcrumb.
                # Since we don't have order here, length is a decent proxy.
                best_category = sorted(candidates, key=len, reverse=True)[0]
        
        # 2. Fallback to 'Main Category' if still no category found
        if not best_category and 'Main Category' in attrs:
            if attrs['Main Category'] and attrs['Main Category'][0]:
                best_category = attrs['Main Category'][0]

        return best_category

        return best_category

# Singleton instance for shared use
_kb_instance = None

def get_kb_instance(kb_dir='/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb'):
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = AttributeKnowledgeBase(kb_dir)
    return _kb_instance
