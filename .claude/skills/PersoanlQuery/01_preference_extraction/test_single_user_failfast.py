#!/usr/bin/env python3
"""单用户测试脚本 - 验证 fail-fast 错误处理"""

import subprocess
import json
import os
from datetime import datetime

USER_ID = "A13OFOB1394G31"
INPUT_FILE = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction"

print(f"[{datetime.now().isoformat()}] 🧪 开始单用户fail-fast测试")
print(f"[{datetime.now().isoformat()}] 用户: {USER_ID}")

if not os.path.exists(INPUT_FILE):
    print(f"[{datetime.now().isoformat()}] ❌ 输入文件不存在: {INPUT_FILE}")
    exit(1)

print(f"[{datetime.now().isoformat()}] 🚀 运行主提取脚本")
cmd = [
    "python3",
    "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences_v2_with_aspects.py",
    "--input-file", INPUT_FILE,
    "--output-dir", OUTPUT_DIR
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"[{datetime.now().isoformat()}] ❌ 脚本执行失败")
    print("STDOUT:", result.stdout[-500:] if result.stdout else "(empty)")
    print("STDERR:", result.stderr[-500:] if result.stderr else "(empty)")
    exit(1)

print(f"[{datetime.now().isoformat()}] ✅ 脚本执行成功")
print(result.stdout)

output_file = os.path.join(OUTPUT_DIR, f"preferences_{USER_ID}_v2.json")
if os.path.exists(output_file):
    with open(output_file) as f:
        data = json.load(f)
    size = os.path.getsize(output_file)
    prefs = len(data.get('target_user_preferences', {}))
    aspects = len(data.get('target_user_aspects', []))
    print(f"[{datetime.now().isoformat()}] ✨ 输出文件验证成功")
    print(f"[{datetime.now().isoformat()}]   文件: {output_file}")
    print(f"[{datetime.now().isoformat()}]   大小: {size/1024:.1f} KB")
    print(f"[{datetime.now().isoformat()}]   维度: {prefs}, 方面: {aspects}")
else:
    print(f"[{datetime.now().isoformat()}] ❌ 输出文件未生成: {output_file}")
    exit(1)
