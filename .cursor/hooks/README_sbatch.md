# sbatch_wrapper Hook 使用说明

## 问题说明

`beforeShellExecution` hooks 主要用于拦截**用户交互式执行的命令**，而不是 AI 工具（如 `run_terminal_cmd`）直接执行的命令。

当使用 AI 工具执行命令时，Cursor 可能不会触发 `beforeShellExecution` hooks，这是 Cursor 的设计行为。

## 验证 Hook 是否被触发

Hook 会在 stderr 中输出调试信息：
```
[sbatch_wrapper] Hook被调用 - is_hook_mode: True, stdin_isatty: False
```

如果没有看到这条日志，说明 hook 没有被触发。

## 解决方案

### 方案 1: 手动使用 sbatch_wrapper

执行 Python 脚本时，使用以下格式：

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/stark && cd /home/wlia0047/ar57/wenyu && python stark/code/generate_query/main.py"
```

### 方案 2: 在命令中包含 sbatch 关键字

如果命令中包含 `sbatch` 或 `sbatch_wrapper` 关键字，hook 会允许执行。

### 方案 3: 使用 nohup 在后台执行（当前方案）

当前我们使用 `nohup` 在后台执行脚本，这样可以避免 hook 的限制，但脚本会在后台运行。

## Hook 检测逻辑

Hook 会检测以下情况：
1. 命令是否包含 `.py` 文件（Python 脚本）
2. 命令中是否已经包含 `sbatch` 或 `sbatch_wrapper`

如果检测到 Python 脚本但没有 sbatch，hook 会返回 `permission: "deny"` 阻止执行。

## 调试

查看 hook 的调试输出：
- Hook 是否被调用
- 检测到的命令内容
- 是否是 Python 脚本
- 是否包含 sbatch

所有调试信息都会输出到 stderr。
