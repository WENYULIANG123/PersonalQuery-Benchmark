#!/usr/bin/env python3
"""包含大量调试日志的单用户提取测试"""

import sys
import os
import json
from datetime import datetime

print(f"[DEBUG {datetime.now().isoformat()}] 脚本启动")
print(f"[DEBUG {datetime.now().isoformat()}] Python版本: {sys.version}")
print(f"[DEBUG {datetime.now().isoformat()}] 工作目录: {os.getcwd()}")

USER_ID = "A13OFOB1394G31"
INPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction"
INPUT_FILE = os.path.join(INPUT_DIR, f"reviews_{USER_ID}.json")

print(f"[DEBUG {datetime.now().isoformat()}] 配置: USER_ID={USER_ID}")
print(f"[DEBUG {datetime.now().isoformat()}] 输入路径: {INPUT_FILE}")
print(f"[DEBUG {datetime.now().isoformat()}] 输出路径: {OUTPUT_DIR}")

print(f"[DEBUG {datetime.now().isoformat()}] 检查输入文件存在性...")
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR {datetime.now().isoformat()}] 输入文件不存在: {INPUT_FILE}")
    sys.exit(1)

file_size = os.path.getsize(INPUT_FILE)
print(f"[DEBUG {datetime.now().isoformat()}] ✅ 输入文件存在, 大小: {file_size} 字节")

print(f"[DEBUG {datetime.now().isoformat()}] 检查输出目录...")
if not os.path.exists(OUTPUT_DIR):
    print(f"[ERROR {datetime.now().isoformat()}] 输出目录不存在: {OUTPUT_DIR}")
    sys.exit(1)

print(f"[DEBUG {datetime.now().isoformat()}] ✅ 输出目录存在")

print(f"[DEBUG {datetime.now().isoformat()}] 加载输入文件...")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ JSON 加载成功")
except Exception as e:
    print(f"[ERROR {datetime.now().isoformat()}] JSON 加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

print(f"[DEBUG {datetime.now().isoformat()}] 检查数据结构...")
print(f"[DEBUG {datetime.now().isoformat()}] 顶级键: {list(data.keys())}")
results = data.get('results', [])
print(f"[DEBUG {datetime.now().isoformat()}] 产品数量: {len(results)}")

if len(results) > 0:
    first = results[0]
    print(f"[DEBUG {datetime.now().isoformat()}] 第一个产品的键: {list(first.keys())}")
    print(f"[DEBUG {datetime.now().isoformat()}] 示例: asin={first.get('asin')}, target_review type={type(first.get('target_review'))}")

print(f"[DEBUG {datetime.now().isoformat()}] 添加技能路径到 sys.path...")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
print(f"[DEBUG {datetime.now().isoformat()}] sys.path: {sys.path[:3]}")

print(f"[DEBUG {datetime.now().isoformat()}] 尝试导入 llm_client...")
try:
    from llm_client import LLMClient
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ llm_client 导入成功")
except Exception as e:
    print(f"[ERROR {datetime.now().isoformat()}] llm_client 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[DEBUG {datetime.now().isoformat()}] 尝试导入提取脚本的主函数...")
try:
    from PersoanlQuery.src.stage_01_preference_extraction import extract_preferences_for_user
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ extract_preferences_for_user 导入成功")
except Exception as e:
    print(f"[DEBUG {datetime.now().isoformat()}] 导入失败 (尝试替代方案): {e}")
    print(f"[DEBUG {datetime.now().isoformat()}] 尝试直接从 01_extract_preferences_v2_with_aspects.py 加载...")

print(f"[DEBUG {datetime.now().isoformat()}] 列出 /home/wlia0047/ar57/wenyu/.claude/skills 目录...")
try:
    skills_dir = "/home/wlia0047/ar57/wenyu/.claude/skills"
    items = os.listdir(skills_dir)
    print(f"[DEBUG {datetime.now().isoformat()}] 目录内容 ({len(items)} 项):")
    for item in sorted(items)[:15]:
        print(f"[DEBUG {datetime.now().isoformat()}]   - {item}")
except Exception as e:
    print(f"[ERROR {datetime.now().isoformat()}] 列出目录失败: {e}")

print(f"[DEBUG {datetime.now().isoformat()}] 列出 PersoanlQuery 目录...")
try:
    pq_dir = "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery"
    if os.path.exists(pq_dir):
        items = os.listdir(pq_dir)
        print(f"[DEBUG {datetime.now().isoformat()}] PersoanlQuery 内容 ({len(items)} 项):")
        for item in sorted(items)[:10]:
            print(f"[DEBUG {datetime.now().isoformat()}]   - {item}")
    else:
        print(f"[ERROR {datetime.now().isoformat()}] PersoanlQuery 目录不存在: {pq_dir}")
except Exception as e:
    print(f"[ERROR {datetime.now().isoformat()}] 列出 PersoanlQuery 失败: {e}")

print(f"[DEBUG {datetime.now().isoformat()}] 检查提取脚本...")
extract_script = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences_v2_with_aspects.py"
if os.path.exists(extract_script):
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ 提取脚本存在")
    size = os.path.getsize(extract_script)
    print(f"[DEBUG {datetime.now().isoformat()}] 脚本大小: {size} 字节")
else:
    print(f"[ERROR {datetime.now().isoformat()}] 提取脚本不存在: {extract_script}")
    sys.exit(1)

print(f"[DEBUG {datetime.now().isoformat()}] 直接执行提取脚本...")
import subprocess
cmd = [
    sys.executable,
    extract_script,
    "--input-file", INPUT_FILE,
    "--output-dir", OUTPUT_DIR
]
print(f"[DEBUG {datetime.now().isoformat()}] 执行命令: {' '.join(cmd[:3])} ...")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ 子进程执行完成, 返回码: {result.returncode}")
    
    if result.stdout:
        print(f"[DEBUG {datetime.now().isoformat()}] STDOUT (最后 500 字):")
        print(result.stdout[-500:])
    else:
        print(f"[DEBUG {datetime.now().isoformat()}] STDOUT: (空)")
        
    if result.stderr:
        print(f"[DEBUG {datetime.now().isoformat()}] STDERR (最后 500 字):")
        print(result.stderr[-500:])
    else:
        print(f"[DEBUG {datetime.now().isoformat()}] STDERR: (空)")
        
except subprocess.TimeoutExpired:
    print(f"[ERROR {datetime.now().isoformat()}] 子进程超时 (>600秒)")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR {datetime.now().isoformat()}] 子进程执行异常: {type(e).__name__}: {e}")
    sys.exit(1)

print(f"[DEBUG {datetime.now().isoformat()}] 检查输出文件...")
output_file = os.path.join(OUTPUT_DIR, f"preferences_{USER_ID}_v2.json")
if os.path.exists(output_file):
    size = os.path.getsize(output_file)
    print(f"[DEBUG {datetime.now().isoformat()}] ✅ 输出文件生成: {output_file} ({size} 字节)")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        prefs = len(output_data.get('target_user_preferences', {}))
        aspects = len(output_data.get('target_user_aspects', []))
        print(f"[DEBUG {datetime.now().isoformat()}] ✅ 输出文件有效")
        print(f"[DEBUG {datetime.now().isoformat()}] 维度: {prefs}, 方面: {aspects}")
    except Exception as e:
        print(f"[ERROR {datetime.now().isoformat()}] 输出文件 JSON 解析失败: {e}")
        sys.exit(1)
else:
    print(f"[ERROR {datetime.now().isoformat()}] 输出文件未生成: {output_file}")
    sys.exit(1)

print(f"[SUCCESS {datetime.now().isoformat()}] ✨ 单用户提取测试完成!")
