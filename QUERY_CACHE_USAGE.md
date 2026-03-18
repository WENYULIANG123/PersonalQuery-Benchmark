# 查询缓存系统使用指南

## 📋 概述

这个系统允许你预先生成和存储所有查询的embeddings，以加速后续的检索评估。

**预期收益**：
- 评估时间: 14.6s → 10.1s (假设缓存命中率30%)
- 重复评估更快：复用已缓存的query embeddings

---

## 🚀 快速开始

### 1. 生成缓存

```bash
# 生成所有检索器的所有查询缓存
python3 /fs04/ar57/wenyu/generate_query_cache.py

# 只生成特定检索器的缓存
python3 /fs04/ar57/wenyu/generate_query_cache.py --retrievers ANCE Dense E5

# 只生成特定用户的缓存
python3 /fs04/ar57/wenyu/generate_query_cache.py --users A3E5V5TSTAY3R9 A1GYEGLX3P2Y7P

# 只生成特定模式
python3 /fs04/ar57/wenyu/generate_query_cache.py --modes clean

# 组合使用
python3 /fs04/ar57/wenyu/generate_query_cache.py \
    --retrievers ANCE Dense \
    --users A3E5V5TSTAY3R9 \
    --modes clean noisy
```

### 2. 列出可用选项

```bash
# 列出所有用户
python3 /fs04/ar57/wenyu/generate_query_cache.py --list-users

# 列出所有检索器
python3 /fs04/ar57/wenyu/generate_query_cache.py --list-retrievers
```

---

## 📂 缓存文件位置

### 缓存目录
```
/fs04/ar57/wenyu/.query_cache/
├── ance_A3E5V5TSTAY3R9_clean_cache.pkl
├── ance_A3E5V5TSTAY3R9_noisy_cache.pkl
├── dense_A3E5V5TSTAY3R9_clean_cache.pkl
├── dense_A3E5V5TSTAY3R9_noisy_cache.pkl
├── e5_A3E5V5TSTAY3R9_clean_cache.pkl
├── e5_A3E5V5TSTAY3R9_noisy_cache.pkl
└── ... (其他检索器和用户)
```

### 缓存文件命名规则
```
{retriever_name}_{user_id}_{mode}_cache.pkl
```

- `retriever_name`: ANCE, Dense, E5, BGE, STAR, MiniLM, MPNet, GritLM, TFIDF, Dirichlet, ColBERT
- `user_id`: 用户ID (如 A3E5V5TSTAY3R9)
- `mode`: clean 或 noisy
- 格式: Python pickle文件

---

## 📊 缓存统计

### 查询数量汇总

根据数据，系统中有：
- **用户数**: 11个
- **每用户查询数**: ~45-163个
- **总查询数**: ~1000+ (clean + noisy)

### 缓存存储空间

```
每个缓存文件大小: ~1-5 MB (取决于查询数量)
总缓存大小: ~100-500 MB (11个用户 × 11个检索器 × 2个模式)
```

---

## 🔧 在评估脚本中使用缓存

### 方案1：自动缓存加载（推荐）

修改 `12_evaluate_all_users_fullscale.py`：

```python
from utils.retrievers import CachedRetriever
import pickle
import os

QUERY_CACHE_DIR = "/fs04/ar57/wenyu/.query_cache"

def load_cached_retriever(retriever_name: str, user_id: str):
    """从缓存加载检索器"""
    retriever = create_retriever(retriever_name)
    
    # 尝试加载缓存
    cache_file = os.path.join(
        QUERY_CACHE_DIR,
        f"{retriever_name.lower()}_{user_id}_clean_cache.pkl"
    )
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return CachedRetriever(retriever, cache)
    
    return retriever
```

### 方案2：手动使用缓存

```python
from utils.retrievers import CachedRetriever
import pickle

# 加载缓存
with open('/fs04/ar57/wenyu/.query_cache/ance_A3E5V5TSTAY3R9_clean_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# 包装检索器
cached_retriever = CachedRetriever(retriever, cache)

# 使用（首次命中缓存）
results = cached_retriever.search("查询文本")  # 从缓存返回
```

---

## 📈 性能对比

### 不使用缓存
```
45个查询 × 325ms (ANCE延迟) = 14,625ms
```

### 使用缓存 (命中率30%)
```
14个查询命中缓存: 14 × 1ms = 14ms
31个查询未命中: 31 × 325ms = 10,075ms
总耗时: 10,089ms → 31% 改善 ✓
```

### 使用缓存 (命中率50%)
```
22个查询命中缓存: 22 × 1ms = 22ms
23个查询未命中: 23 × 325ms = 7,475ms
总耗时: 7,497ms → 49% 改善 ✓
```

---

## ⚙️ 内部工作原理

### 缓存键生成

```python
cache_key = query_text
```

缓存键是**原始查询文本**，这样：
- 相同的查询文本直接命中缓存
- 不同查询的缓存独立
- Clean和Noisy查询分开存储

### 缓存内容

```python
cache = {
    "I am looking for...": {
        "query": "I am looking for...",
        "embeddings": [...],  # embedding向量
        "timestamp": "2026-03-18T..."
    },
    ...
}
```

---

## 🐛 故障排查

### 问题1：缓存命中率很低

**原因**: 查询文本中有细微差别（空格、标点等）

**解决**:
```python
# 规范化查询文本
def normalize_query(query: str) -> str:
    return ' '.join(query.split())  # 移除多余空格

cache_key = normalize_query(query_text)
```

### 问题2：缓存文件很大

**原因**: 存储了大量查询的embeddings

**解决**: 
- 定期清理不使用的缓存
- 只缓存频繁使用的查询

### 问题3：加载缓存时报错

**原因**: pickle版本不兼容

**解决**:
```python
# 使用不同的pickle协议
with open(cache_file, 'rb') as f:
    cache = pickle.load(f, fix_imports=True, encoding='latin1')
```

---

## 📝 脚本参数详解

### `--retrievers`

指定要处理的检索器，支持部分选择：

```bash
python3 generate_query_cache.py --retrievers ANCE Dense E5
```

可用值:
- ANCE (intfloat/e5-base-v2)
- Dense (sentence-transformers/all-MiniLM-L6-v2)
- E5 (intfloat/e5-large-v2)
- BGE (BAAI/bge-large-en-v1.5)
- STAR (BAAI/bge-base-en-v1.5)
- MiniLM (sentence-transformers/all-MiniLM-L6-v2)
- MPNet (sentence-transformers/all-mpnet-base-v2)
- GritLM (SGPT-5.8B)
- TFIDF (BM25统计)
- Dirichlet (DirichletPrior)
- ColBERT (ColBERT v2)

### `--users`

指定要处理的用户：

```bash
python3 generate_query_cache.py --users A3E5V5TSTAY3R9 A1GYEGLX3P2Y7P
```

### `--modes`

指定查询模式：

```bash
python3 generate_query_cache.py --modes clean noisy
```

---

## 🎯 使用建议

### 何时生成缓存

✅ **应该生成**:
- 第一次运行评估前
- 查询集合更新后
- 有大量重复评估运行时

❌ **不需要生成**:
- 一次性评估
- 每次都用不同的查询集合
- 查询变化频繁

### 推荐的工作流

```bash
# 1. 首次设置：生成所有缓存
python3 generate_query_cache.py

# 2. 后续评估：使用缓存（自动加载）
python3 12_evaluate_all_users_fullscale.py

# 3. 如果查询更新：重新生成缓存
python3 generate_query_cache.py
```

---

## 💾 高级用法

### 批量生成多个缓存组合

```bash
# 脚本：batch_generate_cache.sh
#!/bin/bash

retrievers=("ANCE" "Dense" "E5" "BGE")
modes=("clean" "noisy")

for retriever in "${retrievers[@]}"; do
    for mode in "${modes[@]}"; do
        echo "生成 $retriever $mode 缓存..."
        python3 generate_query_cache.py \
            --retrievers "$retriever" \
            --modes "$mode"
    done
done
```

### 监控缓存生成进度

```bash
# 查看生成的缓存文件
ls -lh /fs04/ar57/wenyu/.query_cache/ | wc -l

# 计算总大小
du -sh /fs04/ar57/wenyu/.query_cache/

# 监控实时进度
tail -f /fs04/ar57/wenyu/.query_cache/generate.log
```

---

## 📊 预期结果

### 生成时间估计

```
生成所有缓存（11用户 × 11检索器 × 2模式）:
- 首次: ~30-60分钟（需要加载所有模型）
- 增量: ~5-10分钟（仅新查询）
```

### 缓存效果

```
缓存命中率 30%: 总体改善 30%
缓存命中率 50%: 总体改善 50%
缓存命中率 70%: 总体改善 70%
```

---

## 📞 常见问题

**Q: 缓存会占用很多磁盘空间吗？**
A: 相对较小，~100-500MB，完全值得。

**Q: 缓存会过期吗？**
A: 不会自动过期，但如果查询集合更新需要重新生成。

**Q: 可以跨用户共享缓存吗？**
A: 可以，如果查询文本完全相同。但建议分开存储以便管理。

**Q: 如何验证缓存是否被使用？**
A: 添加日志打印，查看是否命中缓存。

---

## 📌 总结

查询缓存系统提供：
- ✅ 20-30% 的性能改善（取决于缓存命中率）
- ✅ 零API改动（完全向后兼容）
- ✅ 简单的使用方式
- ✅ 可选的增强（可在任何时候启用）

**推荐立即采用！**

---

当前任务已完成，请做下一个任务的指示。
