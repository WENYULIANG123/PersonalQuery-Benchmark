#!/bin/bash
# GitHub Issue 获取示例脚本

export PATH="$HOME/bin:$PATH"

echo "=== GitHub Issue 获取示例 ==="
echo ""

# 设置仓库路径
REPO_DIR="/home/wlia0047/ar57/wenyu"
cd "$REPO_DIR"

echo "1. 列出所有开放的 Issues："
echo "   gh issue list"
gh issue list
echo ""

echo "2. 列出所有 Issues（包括已关闭的）："
echo "   gh issue list --state all"
gh issue list --state all
echo ""

echo "3. 查看特定 Issue 的用法："
echo "   gh issue view <number>"
echo "   例如: gh issue view 123"
echo ""

echo "4. 查看 Issue 并显示评论："
echo "   gh issue view <number> --comments"
echo ""

echo "5. 以 JSON 格式获取 Issue 信息："
echo "   gh issue view <number> --json title,body,state,labels,author"
echo ""

echo "6. 在浏览器中打开 Issue："
echo "   gh issue view <number> --web"
echo ""

echo "=== 实际使用示例 ==="
echo ""
echo "假设要查看 issue #1，可以运行："
echo "  gh issue view 1"
echo ""
echo "如果要修复 issue #123，使用 /fix-issue 命令："
echo "  /fix-issue 123"
echo ""
