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
log_msg("🔬 PersonalQuery - 完整测试（所有三个方案）")
log_msg("="*80)

with open(REVIEW_FILE) as f:
    data = json.load(f)
products = data.get('results', [])

log_msg(f"📊 输入: {REVIEW_FILE}")
log_msg(f"📊 用户: {TEST_USER_ID}")
log_msg(f"📊 产品数: {len(products)}")
log_msg(f"📊 输出目录: {OUTPUT_DIR}")

tests = [
    {
        "name": "方案 B (v2) - 维度+方面双视角⭐",
        "script": "01_extract_preferences_v2_with_aspects.py",
        "args": f"--input-file {REVIEW_FILE} --output-dir {OUTPUT_DIR}",
        "expected_output": f"{OUTPUT_DIR}/preferences_{TEST_USER_ID}_v2.json",
        "timeout": 7200
    },
    {
        "name": "方案 C (Template1) - 方面提取",
        "script": "01_aspect_extraction.py",
        "args": f"--input-file {REVIEW_FILE} --output-dir {OUTPUT_DIR}",
        "expected_output": f"{OUTPUT_DIR}/aspects_{TEST_USER_ID}.json",
        "timeout": 7200
    },
    {
        "name": "方案 C (Template2) - 方面整合",
        "script": "01_aspect_consolidation.py",
        "args": f"--input-file {OUTPUT_DIR}/aspects_{TEST_USER_ID}.json --output-dir {OUTPUT_DIR}",
        "expected_output": f"{OUTPUT_DIR}/consolidated_aspects_{TEST_USER_ID}.json",
        "timeout": 3600
    }
]

results = {}

for i, test_cfg in enumerate(tests, 1):
    log_msg(f"\n{'='*80}")
    log_msg(f"[{i}/3] 🚀 {test_cfg['name']}")
    log_msg(f"{'='*80}")
    
    script_path = os.path.join(IMPL_DIR, test_cfg['script'])
    cmd = f"""python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \\
        "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \\
         conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \\
         cd {IMPL_DIR} && \\
         python3 {test_cfg['script']} {test_cfg['args']}\""""
    
    start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=test_cfg['timeout'])
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log_msg(f"✅ 成功 | 耗时: {elapsed:.1f}s ({elapsed/60:.1f}m)")
            results[test_cfg['name']] = {"status": "success", "elapsed": elapsed}
        else:
            log_msg(f"❌ 失败 (exit code {result.returncode}) | 耗时: {elapsed:.1f}s")
            if result.stderr:
                log_msg(f"❌ 错误输出: {result.stderr[:500]}")
            results[test_cfg['name']] = {"status": "failed", "elapsed": elapsed}
        
        if result.stdout:
            log_msg(f"📝 输出摘要 (最后 500 字):\n{result.stdout[-500:]}")
    
    except subprocess.TimeoutExpired:
        log_msg(f"❌ 超时 (>{test_cfg['timeout']}s)")
        results[test_cfg['name']] = {"status": "timeout", "elapsed": test_cfg['timeout']}
    except Exception as e:
        log_msg(f"❌ 异常: {str(e)[:200]}")
        results[test_cfg['name']] = {"status": "error", "error": str(e)[:200]}

log_msg(f"\n{'='*80}")
log_msg("📊 输出验证")
log_msg(f"{'='*80}")

for test_cfg in tests:
    output_file = test_cfg['expected_output']
    name = test_cfg['name']
    
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                data = json.load(f)
            
            size = os.path.getsize(output_file)
            
            if 'preferences' in data:
                count = len(data['preferences'])
                log_msg(f"✅ {name}")
                log_msg(f"   文件: {size} 字节")
                log_msg(f"   偏好数: {count}")
            elif 'target_aspects' in data:
                count = len(data.get('target_aspects', []))
                log_msg(f"✅ {name}")
                log_msg(f"   文件: {size} 字节")
                log_msg(f"   方面数: {count}")
            elif 'consolidated_aspects' in data:
                count = len(data.get('consolidated_aspects', []))
                log_msg(f"✅ {name}")
                log_msg(f"   文件: {size} 字节")
                log_msg(f"   整合方面数: {count}")
            else:
                log_msg(f"⚠️  {name}: 文件存在但格式未知")
        except Exception as e:
            log_msg(f"❌ {name}: 无法解析 JSON - {str(e)[:100]}")
    else:
        log_msg(f"❌ {name}: 文件不存在 - {output_file}")

log_msg(f"\n{'='*80}")
log_msg("🎯 测试摘要")
log_msg(f"{'='*80}")

for name, result in results.items():
    status = result['status']
    elapsed = result.get('elapsed', 0)
    log_msg(f"{name:40} {status:10} {elapsed:7.1f}s")

log_msg(f"\n✅ 测试完成！")
