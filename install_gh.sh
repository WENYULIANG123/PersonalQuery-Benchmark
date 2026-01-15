#!/bin/bash
# GitHub CLI 安装脚本

set -e

echo "=== GitHub CLI 安装脚本 ==="
echo ""

# 检查是否已有 gh
if command -v gh &> /dev/null; then
    echo "✓ GitHub CLI 已安装: $(gh --version | head -1)"
    exit 0
fi

# 方法1: 使用 dnf 安装（需要 sudo）
echo "方法1: 尝试使用 dnf 安装（需要 sudo 权限）..."
if command -v sudo &> /dev/null; then
    echo "安装 dnf config-manager 插件..."
    sudo dnf install -y 'dnf-command(config-manager)' || echo "跳过..."
    
    echo "添加 GitHub CLI 仓库..."
    sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo || echo "仓库可能已存在"
    
    echo "安装 GitHub CLI..."
    if sudo dnf install -y gh; then
        echo "✓ 安装成功！"
        gh --version
        exit 0
    else
        echo "✗ dnf 安装失败，尝试其他方法..."
    fi
else
    echo "✗ 未找到 sudo 命令，跳过 dnf 安装"
fi

# 方法2: 下载二进制文件到用户目录
echo ""
echo "方法2: 下载二进制文件到 ~/bin..."
mkdir -p ~/bin

# 获取最新版本
LATEST_VERSION=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' | sed 's/v//')
echo "最新版本: $LATEST_VERSION"

# 下载
DOWNLOAD_URL="https://github.com/cli/cli/releases/download/v${LATEST_VERSION}/gh_${LATEST_VERSION}_linux_amd64.tar.gz"
echo "下载地址: $DOWNLOAD_URL"

cd /tmp
curl -L -o gh.tar.gz "$DOWNLOAD_URL" || {
    echo "✗ 下载失败"
    exit 1
}

# 解压
tar -xzf gh.tar.gz
mv gh_${LATEST_VERSION}_linux_amd64/bin/gh ~/bin/gh
chmod +x ~/bin/gh
rm -rf gh_${LATEST_VERSION}_linux_amd64 gh.tar.gz

# 添加到 PATH（如果还没有）
if ! echo "$PATH" | grep -q "$HOME/bin"; then
    echo ""
    echo "⚠️  需要将 ~/bin 添加到 PATH"
    echo "请运行以下命令之一："
    echo "  echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bashrc"
    echo "  或者"
    echo "  echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bash_profile"
    echo ""
    echo "然后运行: source ~/.bashrc 或 source ~/.bash_profile"
fi

# 验证安装
if ~/bin/gh --version &> /dev/null; then
    echo "✓ 安装成功！"
    ~/bin/gh --version
    echo ""
    echo "使用前请确保 ~/bin 在 PATH 中，或使用完整路径: ~/bin/gh"
else
    echo "✗ 安装验证失败"
    exit 1
fi
