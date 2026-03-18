#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import time
from datetime import datetime

TEST_USER_ID = "ALYZJ7W14YS26"
REVIEW_FILE = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{TEST_USER_ID}.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/test_outputs"
IMPL_DIR = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction"

def log_msg(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_msg("\n" + "="*80)
log_msg("🔬 PersonalQuery Stage 1 - 实际测试（修复版）")
log_msg("="*80)

log_msg(f"📝 输入文件: {REVIEW_FILE}")
log_msg(f"📝 输出目录: {OUTPUT_DIR}")
log_msg(f"📝 并发数: 5 (降低以避免限流)")

tests = [
    ("Template 1 (方面提取)", "01_aspect_extraction.py", f"--input-file {REVIEW_FILE} --output-dir {OUTPUT_DIR} --max-workers 5"),
    ("Template 2 (方面整合)", "01_aspect_consolidation.py", f"--input-file {OUTPUT_DIR}/aspects_{TEST_USER_ID}.json --output-dir {OUTPUT_DIR}"),
]

results = {}

for name, script, args in tests:
    log_msg(f"\n🚀 运行: {name}")
    cmd = f"""
    python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \\
        "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \\
         conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \\
         cd {IMPL_DIR} && \\
         python3 {script} {args}"
    """
    
    start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log_msg(f"✅ {name} 成功 ({elapsed:.1f}s)")
            results[name] = "success"
        else:
            log_msg(f"❌ {name} 失败")
            print(f"STDERR: {result.stderr[:300]}")
            results[name] = "failed"
    except Exception as e:
        log_msg(f"❌ {name} 异常: {str(e)[:100]}")
        results[name] = "error"

log_msg(f"\n{'='*80}")
log_msg("📊 检查输出文件...")

output_files = {
    "Template 1": f"{OUTPUT_DIR}/aspects_{TEST_USER_ID}.json",
    "Template 2": f"{OUTPUT_DIR}/consolidated_aspects_{TEST_USER_ID}.json"
}

for name, filepath in output_files.items():
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        with open(filepath) as f:
            data = json.load(f)
        
        if 'aspects' in data:
            count = len(data['aspects'])
            log_msg(f"✅ {name}: {size} 字节，{count} 个方面")
        elif 'consolidated_aspects' in data:
            count = len(data['consolidated_aspects'])
            log_msg(f"✅ {name}: {size} 字节，{count} 个整合方面")
        else:
            log_msg(f"⚠️  {name}: 文件存在但格式未知")
    else:
        log_msg(f"❌ {name}: 文件不存在")

log_msg(f"\n{'='*80}")
log_msg("🎯 测试完成")
for name, status in results.items():
    log_msg(f"  {name:30} {status}")
