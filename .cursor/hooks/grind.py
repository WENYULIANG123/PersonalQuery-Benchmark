#!/usr/bin/env python3
"""
长时间运行的 Agent 循环 Hook
用于创建可长时间运行的 Agent，让其不断迭代，直到达成目标。

使用场景：
- 反复运行并修复，直到所有测试通过
- 不断迭代 UI，直到与设计稿完全匹配
- 任何结果可验证的目标导向任务
"""

import json
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# 配置常量
MAX_ITERATIONS = 5
SCRATCHPAD_PATH = Path(".cursor/scratchpad.md")
DONE_MARKER = "DONE"

# 自动完成检测配置
AUTO_DETECT_ENABLED = True  # 启用自动检测
AUTO_MARK_DONE_ON_MAX = True  # 达到最大迭代次数时自动添加 DONE
AUTO_MARK_DONE_ON_SUCCESS_KEYWORDS = True  # 检测成功关键词时自动添加 DONE
AUTO_OUTPUT_SUCCESS_KEYWORDS = True  # 任务完成时自动输出成功关键词到 scratchpad

# 成功关键词列表（在 scratchpad 中检测这些关键词）
SUCCESS_KEYWORDS = [
    r"所有测试通过",
    r"全部测试通过",
    r"测试全部通过",
    r"任务完成",
    r"任务已完成",
    r"已完成",
    r"✓.*完成",
    r"所有.*通过",
    r"成功.*完成",
    r"TASK.*COMPLETED",
    r"SUCCESS",
]

# 成功标志文件路径（如果存在这些文件，则认为任务完成）
SUCCESS_MARKER_FILES = [
    Path(".cursor/.task_completed"),
    Path(".task_success"),
    Path(".success"),
]

# 自动输出的成功关键词内容模板（任务完成时自动添加到 scratchpad）
SUCCESS_KEYWORDS_OUTPUT_TEMPLATE = """## 任务完成状态

✓ 任务已完成
✓ 所有测试通过
✓ 任务完成

**完成时间**: {timestamp}
**迭代次数**: {loop_count}/{max_iterations}

任务完成！
"""


def read_stdin() -> Dict[str, Any]:
    """从标准输入读取 JSON 数据"""
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}", file=sys.stderr)
        return {}


def check_scratchpad() -> bool:
    """检查 scratchpad 文件中是否包含完成标记"""
    if not SCRATCHPAD_PATH.exists():
        return False
    
    try:
        content = SCRATCHPAD_PATH.read_text(encoding="utf-8")
        return DONE_MARKER in content
    except Exception as e:
        print(f"Error reading scratchpad: {e}", file=sys.stderr)
        return False


def read_scratchpad() -> str:
    """读取 scratchpad 文件内容"""
    if not SCRATCHPAD_PATH.exists():
        return ""
    
    try:
        return SCRATCHPAD_PATH.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading scratchpad: {e}", file=sys.stderr)
        return ""


def write_scratchpad(content: str) -> bool:
    """写入内容到 scratchpad 文件"""
    try:
        SCRATCHPAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCRATCHPAD_PATH.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error writing scratchpad: {e}", file=sys.stderr)
        return False


def append_to_scratchpad(text: str) -> bool:
    """追加内容到 scratchpad 文件"""
    try:
        SCRATCHPAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        current_content = read_scratchpad()
        new_content = current_content + "\n" + text if current_content else text
        return write_scratchpad(new_content)
    except Exception as e:
        print(f"Error appending to scratchpad: {e}", file=sys.stderr)
        return False


def check_success_keywords(content: str) -> bool:
    """检查内容中是否包含成功关键词"""
    if not AUTO_MARK_DONE_ON_SUCCESS_KEYWORDS:
        return False
    
    for keyword in SUCCESS_KEYWORDS:
        try:
            if re.search(keyword, content, re.IGNORECASE | re.MULTILINE):
                print(f"[Grind Hook] 检测到成功关键词: {keyword}", file=sys.stderr)
                return True
        except re.error as e:
            print(f"Error matching keyword {keyword}: {e}", file=sys.stderr)
    
    return False


def check_success_marker_files() -> bool:
    """检查是否存在成功标志文件"""
    for marker_file in SUCCESS_MARKER_FILES:
        if marker_file.exists():
            print(f"[Grind Hook] 检测到成功标志文件: {marker_file}", file=sys.stderr)
            return True
    return False


def detect_task_completion(input_data: Dict[str, Any]) -> Optional[str]:
    """
    自动检测任务是否完成
    
    返回:
        None: 未检测到完成条件
        str: 完成原因
    """
    if not AUTO_DETECT_ENABLED:
        return None
    
    status = input_data.get("status", "")
    loop_count = input_data.get("loop_count", 0)
    
    # 检查是否达到最大迭代次数
    if AUTO_MARK_DONE_ON_MAX and loop_count >= MAX_ITERATIONS - 1:
        # 当前是最后一次迭代，完成后自动添加 DONE
        return "已达到最大迭代次数"
    
    # 检查 scratchpad 中的成功关键词
    scratchpad_content = read_scratchpad()
    if scratchpad_content and check_success_keywords(scratchpad_content):
        return "检测到成功关键词"
    
    # 检查成功标志文件
    if check_success_marker_files():
        return "检测到成功标志文件"
    
    # 检查状态信息中是否有成功提示
    followup_message = input_data.get("followup_message", "")
    if followup_message and check_success_keywords(followup_message):
        return "在消息中检测到成功关键词"
    
    return None


def auto_output_success_keywords(loop_count: int) -> bool:
    """自动输出成功关键词到 scratchpad"""
    if not AUTO_OUTPUT_SUCCESS_KEYWORDS:
        return False
    
    try:
        current_content = read_scratchpad()
        
        # 如果已经包含成功关键词，则不需要再次输出
        if check_success_keywords(current_content):
            print(f"[Grind Hook] 成功关键词已存在，跳过输出", file=sys.stderr)
            return True
        
        # 获取当前时间戳
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except:
            timestamp = "未知时间"
        
        # 格式化成功关键词内容
        success_content = SUCCESS_KEYWORDS_OUTPUT_TEMPLATE.format(
            timestamp=timestamp,
            loop_count=loop_count + 1,
            max_iterations=MAX_ITERATIONS
        )
        
        # 追加到 scratchpad
        new_content = current_content + "\n\n" + success_content
        
        if write_scratchpad(new_content):
            print(f"[Grind Hook] 已自动输出成功关键词到 scratchpad", file=sys.stderr)
            return True
        else:
            print(f"[Grind Hook] 无法自动输出成功关键词", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[Grind Hook] 自动输出成功关键词时出错: {e}", file=sys.stderr)
        return False


def auto_mark_done(reason: str) -> bool:
    """自动添加 DONE 标记到 scratchpad"""
    try:
        current_content = read_scratchpad()
        
        # 如果已经包含 DONE 标记，则不需要再次添加
        if DONE_MARKER in current_content:
            print(f"[Grind Hook] DONE 标记已存在，跳过添加", file=sys.stderr)
            return True
        
        # 添加 DONE 标记，并附带原因
        done_section = f"\n\n## 自动完成标记\n\n**完成原因**: {reason}\n**完成时间**: {os.popen('date').read().strip()}\n\n```\n{DONE_MARKER}\n```\n"
        
        new_content = current_content + done_section
        
        if write_scratchpad(new_content):
            print(f"[Grind Hook] 已自动添加 DONE 标记 (原因: {reason})", file=sys.stderr)
            return True
        else:
            print(f"[Grind Hook] 无法自动添加 DONE 标记", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[Grind Hook] 自动添加 DONE 标记时出错: {e}", file=sys.stderr)
        return False


def should_continue(input_data: Dict[str, Any]) -> bool:
    """判断是否应该继续循环"""
    status = input_data.get("status", "")
    loop_count = input_data.get("loop_count", 0)
    
    # 如果状态不是 completed，或者达到最大迭代次数，则停止
    if status != "completed":
        return False
    
    if loop_count >= MAX_ITERATIONS:
        # 达到最大迭代次数，先输出成功关键词，然后自动添加 DONE
        if AUTO_OUTPUT_SUCCESS_KEYWORDS:
            auto_output_success_keywords(loop_count)
        if AUTO_MARK_DONE_ON_MAX and AUTO_DETECT_ENABLED:
            auto_mark_done("达到最大迭代次数")
        return False
    
    # 检查 scratchpad 中是否有完成标记
    if check_scratchpad():
        return False
    
    # 任务完成时，先自动输出成功关键词（如果启用）
    if AUTO_OUTPUT_SUCCESS_KEYWORDS and status == "completed":
        auto_output_success_keywords(loop_count)
        # 输出后重新读取 scratchpad，检查是否包含成功关键词
        scratchpad_content = read_scratchpad()
        if scratchpad_content and check_success_keywords(scratchpad_content):
            # 检测到成功关键词，自动添加 DONE 标记
            auto_mark_done("检测到成功关键词（自动输出）")
            return False
    
    # 自动检测任务完成条件
    completion_reason = detect_task_completion(input_data)
    if completion_reason:
        print(f"[Grind Hook] 检测到任务完成条件: {completion_reason}", file=sys.stderr)
        auto_mark_done(completion_reason)
        return False
    
    return True


def main():
    """主函数"""
    input_data = read_stdin()
    
    # 提取关键信息
    status = input_data.get("status", "unknown")
    loop_count = input_data.get("loop_count", 0)
    conversation_id = input_data.get("conversation_id", "")
    
    # 输出调试信息到 stderr（不会影响 JSON 输出）
    print(f"[Grind Hook] Status: {status}, Loop: {loop_count}/{MAX_ITERATIONS}", 
          file=sys.stderr)
    
    # 判断是否应该继续
    if not should_continue(input_data):
        # 返回空对象表示停止循环
        print(json.dumps({}))
        return
    
    # 继续循环
    next_iteration = loop_count + 1
    
    # 如果是最后一次迭代，提示 Agent 添加 DONE 标记
    if next_iteration >= MAX_ITERATIONS:
        followup_message = (
            f"[迭代 {next_iteration}/{MAX_ITERATIONS}] 继续执行（最后一次迭代）。"
            f"完成后在 {SCRATCHPAD_PATH} 中更新 {DONE_MARKER}，或系统将自动标记完成。"
        )
    else:
        followup_message = (
            f"[迭代 {next_iteration}/{MAX_ITERATIONS}] "
            f"继续执行。完成后可以在 {SCRATCHPAD_PATH} 中更新 {DONE_MARKER}，"
            f"或者在 scratchpad 中包含成功关键词（如'任务完成'、'所有测试通过'等）以自动停止。"
        )
    
    response = {
        "followup_message": followup_message
    }
    
    print(json.dumps(response))


if __name__ == "__main__":
    main()
