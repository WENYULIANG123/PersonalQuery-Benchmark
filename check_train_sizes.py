import json
from pathlib import Path

loocv_data_dir = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data')

print("TRAINING DATA SIZES")
print("="*80)
for user_dir in sorted(loocv_data_dir.glob('user_*')):
    user_id = user_dir.name.replace('user_', '')
    
    personal_train_file = user_dir / 'personal_train.json'
    global_train_file = user_dir / 'global_train.json'
    holdout_file = user_dir / 'holdout.json'
    
    with open(personal_train_file) as f:
        personal_train = json.load(f).get('pairs', [])
    
    with open(global_train_file) as f:
        global_train = json.load(f).get('pairs', [])
    
    with open(holdout_file) as f:
        holdout = json.load(f).get('pairs', [])
    
    print(f"{user_id:20s} | Personal: {len(personal_train):4d} | Global: {len(global_train):4d} | Holdout: {len(holdout):4d}")

