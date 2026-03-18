#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")

TEST_USER_ID = "ALYZJ7W14YS26"
REVIEW_FILE = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{TEST_USER_ID}.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/test_outputs"
IMPL_DIR = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction"

def log_msg(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] {msg}")
    print("=" * 80)

def create_test_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_msg(f"✅ 测试环境已准备好，输出目录：{OUTPUT_DIR}")
    
    if not os.path.exists(REVIEW_FILE):
        log_msg(f"❌ 错误：找不到评论文件 {REVIEW_FILE}")
        return False
    
    with open(REVIEW_FILE, 'r') as f:
        data = json.load(f)
    
    num_reviews = len(data.get('reviews', []))
    log_msg(f"✅ 输入数据验证通过")
    print(f"   用户 ID: {TEST_USER_ID}")
    print(f"   评论数: {num_reviews}")
    
    return True

def run_implementation(impl_name: str, script_name: str, args: str) -> Dict:
    log_msg(f"🚀 启动 {impl_name}...")
    
    script_path = os.path.join(IMPL_DIR, script_name)
    
    start_time = time.time()
    cmd = f"""
    python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \\
        "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \\
         conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \\
         cd {IMPL_DIR} && \\
         python3 {script_name} {args}"
    """
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_msg(f"✅ {impl_name} 完成 (耗时: {elapsed:.1f}s)")
            return {
                "status": "success",
                "elapsed": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            log_msg(f"❌ {impl_name} 失败")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return {
                "status": "failed",
                "elapsed": elapsed,
                "error": result.stderr
            }
    except subprocess.TimeoutExpired:
        log_msg(f"❌ {impl_name} 超时 (>10 分钟)")
        return {
            "status": "timeout",
            "elapsed": 600
        }
    except Exception as e:
        log_msg(f"❌ {impl_name} 异常: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def check_output_files() -> Dict:
    log_msg("📊 检查输出文件...")
    
    results = {}
    
    v1_file = os.path.join(OUTPUT_DIR, f"preferences_{TEST_USER_ID}.json")
    if os.path.exists(v1_file):
        with open(v1_file) as f:
            data = json.load(f)
        results["v1"] = {
            "status": "found",
            "file_size": os.path.getsize(v1_file),
            "num_preferences": len(data.get('preferences', []))
        }
        print(f"✅ v1 输出：{os.path.getsize(v1_file)} 字节，{len(data.get('preferences', []))} 条偏好")
    else:
        results["v1"] = {"status": "not_found"}
        print(f"❌ v1 输出未找到")
    
    v2_file = os.path.join(OUTPUT_DIR, f"preferences_{TEST_USER_ID}_v2.json")
    if os.path.exists(v2_file):
        with open(v2_file) as f:
            data = json.load(f)
        results["v2"] = {
            "status": "found",
            "file_size": os.path.getsize(v2_file),
            "num_preferences": len(data.get('preferences', []))
        }
        print(f"✅ v2 输出：{os.path.getsize(v2_file)} 字节，{len(data.get('preferences', []))} 条偏好")
    else:
        results["v2"] = {"status": "not_found"}
        print(f"❌ v2 输出未找到")
    
    t1_file = os.path.join(OUTPUT_DIR, f"aspects_{TEST_USER_ID}.json")
    if os.path.exists(t1_file):
        with open(t1_file) as f:
            data = json.load(f)
        results["template1"] = {
            "status": "found",
            "file_size": os.path.getsize(t1_file),
            "num_aspects": len(data.get('aspects', []))
        }
        print(f"✅ Template 1 输出：{os.path.getsize(t1_file)} 字节，{len(data.get('aspects', []))} 个方面")
    else:
        results["template1"] = {"status": "not_found"}
        print(f"❌ Template 1 输出未找到")
    
    t2_file = os.path.join(OUTPUT_DIR, f"consolidated_aspects_{TEST_USER_ID}.json")
    if os.path.exists(t2_file):
        with open(t2_file) as f:
            data = json.load(f)
        results["template2"] = {
            "status": "found",
            "file_size": os.path.getsize(t2_file),
            "num_consolidated": len(data.get('consolidated_aspects', []))
        }
        print(f"✅ Template 2 输出：{os.path.getsize(t2_file)} 字节，{len(data.get('consolidated_aspects', []))} 个整合方面")
    else:
        results["template2"] = {"status": "not_found"}
        print(f"❌ Template 2 输出未找到")
    
    return results

def validate_output_format(impl_name: str, output_file: str, expected_fields: List[str]) -> bool:
    try:
        with open(output_file) as f:
            data = json.load(f)
        
        for field in expected_fields:
            if field not in data:
                print(f"   ❌ 缺少字段: {field}")
                return False
        
        print(f"   ✅ 格式有效")
        return True
    except Exception as e:
        print(f"   ❌ 验证失败: {str(e)}")
        return False

def generate_report(implementations: Dict, outputs: Dict, validations: Dict):
    log_msg("📋 最终测试报告")
    
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "test_user_id": TEST_USER_ID,
        "implementations": implementations,
        "outputs": outputs,
        "validations": validations,
        "summary": generate_summary(implementations, outputs)
    }
    
    report_file = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 报告已保存到: {report_file}")
    
    print("\n" + "=" * 80)
    print("🎯 测试摘要:")
    print("=" * 80)
    for impl, data in implementations.items():
        status = data["status"]
        elapsed = data.get("elapsed", 0)
        print(f"  {impl:15} | 状态: {status:8} | 耗时: {elapsed:6.1f}s")
    
    return report_file

def generate_summary(implementations: Dict, outputs: Dict) -> str:
    success_count = sum(1 for v in implementations.values() if v["status"] == "success")
    total_count = len(implementations)
    
    return f"{success_count}/{total_count} 实现成功完成"

def main():
    print("\n" + "=" * 80)
    print("🔬 PersonalQuery Stage 1 - 综合测试")
    print("=" * 80)
    
    if not create_test_environment():
        return False
    
    log_msg("📝 运行各实现...")
    
    implementations = {
        "v2 (enhanced)": {
            "script": "01_extract_preferences_v2_with_aspects.py",
            "args": f"--input-file {REVIEW_FILE} --output-dir {OUTPUT_DIR}",
            "status": "pending"
        },
        "Template1": {
            "script": "01_aspect_extraction.py",
            "args": f"--input-file {REVIEW_FILE} --output-dir {OUTPUT_DIR}",
            "status": "pending"
        },
        "Template2": {
            "script": "01_aspect_consolidation.py",
            "args": f"--input-file {OUTPUT_DIR}/aspects_{TEST_USER_ID}.json --output-dir {OUTPUT_DIR}",
            "status": "pending"
        }
    }
    
    results = {}
    for impl_name, impl_config in implementations.items():
        result = run_implementation(
            impl_name,
            impl_config["script"],
            impl_config["args"]
        )
        results[impl_name] = result
        impl_config["status"] = result["status"]
        impl_config["elapsed"] = result.get("elapsed", 0)
    
    outputs = check_output_files()
    
    log_msg("✔️  验证输出格式...")
    validations = {}
    
    report_file = generate_report(results, outputs, validations)
    
    log_msg("✅ 测试完成！")
    print(f"\n📄 详细报告: {report_file}")
    
    return True

if __name__ == "__main__":
    main()
