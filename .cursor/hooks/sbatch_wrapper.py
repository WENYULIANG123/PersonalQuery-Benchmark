#!/usr/bin/env python3
# Wrapper script to execute Python commands via sbatch and monitor logs

import sys
import os
import re
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def parse_command(args):
    """Parse command arguments, handling quoted strings like bash script."""
    if not args:
        return None
    
    if len(args) == 1:
        cmd = args[0]
        # Remove outer quotes if present
        if (cmd.startswith("'") and cmd.endswith("'")) or \
           (cmd.startswith('"') and cmd.endswith('"')):
            return cmd[1:-1]
        return cmd
    else:
        # Combine all arguments
        return ' '.join(args)


def is_job_running(job_id):
    """Check if SLURM job is still running."""
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h'],
            capture_output=True,
            timeout=5,
            text=True
        )
        # If squeue returns empty output (no header), job is not in queue
        # If it has output, job is running
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        # If squeue fails or times out, assume job might still be running
        return True
    except Exception:
        return True


def monitor_logs(log_file, err_file, job_id, max_wait=30):
    """Monitor job status until it completes, without outputting log content to terminal."""
    log_path = Path(log_file)
    err_path = Path(err_file)
    
    # Wait for log files to be created (with timeout)
    wait_count = 0
    while wait_count < max_wait:
        if log_path.exists() or err_path.exists():
            break
        # Check job status while waiting
        if not is_job_running(job_id):
            print("[sbatch_wrapper] ⚠️  作业已完成但日志文件尚未创建", file=sys.stderr)
            return
        time.sleep(1)
        wait_count += 1
    
    # Monitor job status without outputting log content
    # Periodically check if job is still running
    last_status_time = 0
    status_interval = 5  # Print status every 5 seconds
    
    try:
        while True:
            # Check if job is still running
            if not is_job_running(job_id):
                # Job is no longer in queue, wait a bit more to ensure it's fully done
                time.sleep(0.5)
                break
            
            # Print status message periodically (not the actual log content)
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                print(f"[sbatch_wrapper] 作业正在运行中 (Job ID: {job_id})...", file=sys.stderr)
                last_status_time = current_time
            
            time.sleep(1)  # Check job status every second
            
    except KeyboardInterrupt:
        # User interrupted, but job continues running
        raise
    except Exception as e:
        print(f"[sbatch_wrapper] ⚠️  监控作业状态时出错: {e}", file=sys.stderr)
        # Continue monitoring even if there's an error
        while True:
            if not is_job_running(job_id):
                break
            time.sleep(1)


def is_python_script_command(command: str) -> bool:
    """Check if command is executing a Python script."""
    if not command:
        return False
    
    command = command.strip()
    
    # Patterns that indicate Python script execution
    patterns = [
        r'^python\s+.*\.py',           # python script.py
        r'^python3\s+.*\.py',           # python3 script.py
        r'\spython\s+.*\.py',           # ... python script.py
        r'\spython3\s+.*\.py',          # ... python3 script.py
        r'\.py\s',                      # .py file with arguments
        r'\.py$',                       # .py file at end
    ]
    
    for pattern in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    
    return False


def has_sbatch_in_command(command: str) -> bool:
    """Check if command already contains sbatch or sbatch_wrapper."""
    if not command:
        return False
    
    command_lower = command.lower()
    script_name = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py"
    
    # Check for various sbatch patterns
    sbatch_patterns = [
        'sbatch',
        'sbatch_wrapper',
        script_name.lower(),
        'sbatch_wrapper.py',
    ]
    
    for pattern in sbatch_patterns:
        if pattern in command_lower or pattern in command:
            return True
    
    return False


def read_input() -> Dict[str, Any]:
    """Read and parse JSON input from stdin (for hook mode)."""
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            if raw_input.strip():
                return json.loads(raw_input)
    except (json.JSONDecodeError, Exception):
        pass
    return {}


def main():
    # Check if running as a hook (has JSON input from stdin)
    input_data = read_input()
    is_hook_mode = bool(input_data)
    
    # Always log when hook is called (even if no input)
    print(f"[sbatch_wrapper] Hook被调用 - is_hook_mode: {is_hook_mode}, stdin_isatty: {sys.stdin.isatty()}", file=sys.stderr)
    
    if is_hook_mode:
        # Hook mode: intercept command and wrap with sbatch
        command = input_data.get("command", "").strip()
        working_dir = (
            input_data.get("working_directory") or
            input_data.get("cwd") or
            input_data.get("workingDirectory") or
            os.getcwd()
        )
        
        # Debug output
        print(f"[sbatch_wrapper] 命令: {command[:200]}", file=sys.stderr)
        print(f"[sbatch_wrapper] 工作目录: {working_dir}", file=sys.stderr)
        
        # Check if this is a Python script command
        is_python_script = is_python_script_command(command)
        has_sbatch = has_sbatch_in_command(command)
        
        print(f"[sbatch_wrapper] 是 Python 脚本命令: {is_python_script}", file=sys.stderr)
        print(f"[sbatch_wrapper] 包含 sbatch: {has_sbatch}", file=sys.stderr)
        
        # If Python script but no sbatch, block execution
        if is_python_script and not has_sbatch:
            script_path = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py"
            print("[sbatch_wrapper] ⚠️  检测到 Python 脚本执行但未使用 sbatch", file=sys.stderr)
            
            output = {
                "continue": True,
                "permission": "deny",
                "user_message": f"Python 脚本必须使用 sbatch 执行。请使用: python3 {script_path} [你的命令]",
                "agent_message": f"检测到 Python 脚本执行但未使用 sbatch。\n重要提示：\n1. Python 脚本必须通过 sbatch 执行\n2. 使用命令: python3 {script_path} {command}\n3. 或者直接使用: python3 {script_path} \"{command}\""
            }
        elif has_sbatch:
            # Already wrapped, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }
        elif command:
            # Not a Python script, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }
        else:
            # No command, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }
        
        # Output JSON response for hook mode
        print(json.dumps(output, ensure_ascii=False))
        return
    
    # Direct mode: execute command via sbatch
    # Parse command from command line arguments
    original_command = parse_command(sys.argv[1:])
    
    if not original_command:
        print("[sbatch_wrapper] ❌ 错误: 未提供要执行的命令", file=sys.stderr)
        sys.exit(1)
    
    # Setup paths
    log_dir = Path("/home/wlia0047/ar57/wenyu/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"sbatch_{timestamp}.log"
    err_file = log_dir / f"sbatch_{timestamp}.err"
    script_file = log_dir / f"sbatch_script_{timestamp}.sh"
    
    # Get current directory
    current_dir = os.getcwd() or "/home/wlia0047/ar57/wenyu"
    
    # Create sbatch script
    script_content = f"""#!/bin/bash
#SBATCH --output={log_file}
#SBATCH --error={err_file}
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
cd "{current_dir}"
# Execute the original command
{original_command}
"""
    
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)
    
    # Submit the job
    print("[sbatch_wrapper] 提交作业到 SLURM...", file=sys.stderr)
    print(f"[sbatch_wrapper] 日志文件: {log_file}", file=sys.stderr)
    print(f"[sbatch_wrapper] 错误文件: {err_file}", file=sys.stderr)
    
    try:
        result = subprocess.run(
            ['sbatch', str(script_file)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from output
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            print("[sbatch_wrapper] ❌ 提交 sbatch 作业失败: 无法解析作业 ID", file=sys.stderr)
            sys.exit(1)
        
        job_id = match.group(1)
        
    except subprocess.CalledProcessError as e:
        print(f"[sbatch_wrapper] ❌ 提交 sbatch 作业失败: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(f"[sbatch_wrapper] ✅ 作业已提交，Job ID: {job_id}", file=sys.stderr)
    print(f"[sbatch_wrapper] 日志文件: {log_file}", file=sys.stderr)
    print(f"[sbatch_wrapper] 错误文件: {err_file}", file=sys.stderr)
    print("[sbatch_wrapper] 开始监控作业状态 (Ctrl+C 停止监控，作业将继续运行)...", file=sys.stderr)
    
    # Monitor logs
    job_completed = False
    try:
        monitor_logs(str(log_file), str(err_file), job_id)
        job_completed = True
    except KeyboardInterrupt:
        print("\n[sbatch_wrapper] 监控已停止，作业将继续在后台运行...", file=sys.stderr)
    
    print(f"[sbatch_wrapper] ✅ 作业已完成 (Job ID: {job_id})", file=sys.stderr)
    if log_file.exists() or err_file.exists():
        print(
            f"[sbatch_wrapper] 查看完整日志: tail -f {log_file} {err_file}",
            file=sys.stderr
        )
    
    # Clean up temporary files after job completion
    if job_completed:
        cleaned_files = []
        failed_files = []
        
        # Clean up script file
        if script_file.exists():
            try:
                script_file.unlink()
                cleaned_files.append(f"脚本文件: {script_file.name}")
            except Exception as e:
                failed_files.append(f"脚本文件: {e}")
        
        # Clean up log file
        if log_file.exists():
            try:
                log_file.unlink()
                cleaned_files.append(f"日志文件: {log_file.name}")
            except Exception as e:
                failed_files.append(f"日志文件: {e}")
        
        # Clean up error file
        if err_file.exists():
            try:
                err_file.unlink()
                cleaned_files.append(f"错误文件: {err_file.name}")
            except Exception as e:
                failed_files.append(f"错误文件: {e}")
        
        # Print cleanup results
        if cleaned_files:
            print(f"[sbatch_wrapper] 🗑️  已清理临时文件: {', '.join(cleaned_files)}", file=sys.stderr)
        if failed_files:
            print(f"[sbatch_wrapper] ⚠️  清理文件失败: {', '.join(failed_files)}", file=sys.stderr)


if __name__ == "__main__":
    main()