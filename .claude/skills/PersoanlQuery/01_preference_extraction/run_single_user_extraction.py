#!/usr/bin/env python3
"""单用户完整提取 - A13OFOB1394G31 的全部 225 个产品"""

import subprocess
import sys
import os
from datetime import datetime

USER_ID = "A13OFOB1394G31"
INPUT_FILE = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction"
SCRIPT = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences_v2_with_aspects.py"

print(f"[{datetime.now().isoformat()}] 🚀 开始单用户完整提取")
print(f"[{datetime.now().isoformat()}] 用户: {USER_ID}")
print(f"[{datetime.now().isoformat()}] 输入: {INPUT_FILE}")
print(f"[{datetime.now().isoformat()}] 输出: {OUTPUT_DIR}")

cmd = [sys.executable, SCRIPT, "--input-file", INPUT_FILE, "--output-dir", OUTPUT_DIR]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode == 0:
        print(f"[{datetime.now().isoformat()}] ✅ 提取成功")
    else:
        print(f"[{datetime.now().isoformat()}] ❌ 提取失败 (exit code: {result.returncode})")
        if result.stderr:
            print(f"[{datetime.now().isoformat()}] 错误:\n{result.stderr[-500:]}")
    
    if result.stdout:
        lines = result.stdout.split('\n')
        print(f"[{datetime.now().isoformat()}] 输出最后 10 行:")
        for line in lines[-10:]:
            if line.strip():
                print(f"[{datetime.now().isoformat()}] {line}")
                
except subprocess.TimeoutExpired:
    print(f"[{datetime.now().isoformat()}] ❌ 提取超时 (>1小时)")
    sys.exit(1)
except Exception as e:
    print(f"[{datetime.now().isoformat()}] ❌ 提取异常: {type(e).__name__}: {e}")
    sys.exit(1)

sys.exit(result.returncode)
