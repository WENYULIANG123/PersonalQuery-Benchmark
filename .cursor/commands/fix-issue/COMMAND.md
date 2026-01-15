# 修复问题命令

修复用户指定的 GitHub issue。

## 步骤

1. 使用 `gh issue view <number>` 获取 issue 详情
2. 在代码库中搜索相关代码
3. 按照现有模式实现修复
4. 如有必要，编写测试
5. 创建引用该 issue 的 PR

## 用法

在 Agent 中输入 `/fix-issue 123` 来修复 issue #123。
