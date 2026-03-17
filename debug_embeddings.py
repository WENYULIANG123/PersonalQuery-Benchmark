import pickle
from pathlib import Path

# Check embeddings files
checkpoints_dir = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints')

# List all embedding files
embedding_files = list(checkpoints_dir.glob('embeddings_*.pkl'))
print(f"Found {len(embedding_files)} embedding files")

if embedding_files:
    # Sample first one
    sample_file = embedding_files[0]
    print(f"\nInspecting: {sample_file.name}")
    
    with open(sample_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"  Type: {type(embeddings)}")
    print(f"  Length: {len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'}")
    
    if isinstance(embeddings, dict):
        print(f"  Keys sample: {list(embeddings.keys())[:5]}")
        
        # Check if values are None
        none_count = sum(1 for v in embeddings.values() if v is None)
        print(f"  None values: {none_count}/{len(embeddings)}")
        
        # Check first non-None value
        for asin, emb in embeddings.items():
            if emb is not None:
                print(f"  Sample embedding shape: {emb.shape if hasattr(emb, 'shape') else type(emb)}")
                break
    
else:
    print("No embedding files found!")
