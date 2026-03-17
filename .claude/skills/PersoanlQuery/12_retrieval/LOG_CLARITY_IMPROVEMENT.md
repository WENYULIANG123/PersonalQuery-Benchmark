# 日志清晰度改进

## 🔍 **发现的问题**

原始日志中，`Processed query` 这一行**没有明确标记**是来自clean还是noisy模式：

```
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (clean mode)...
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (noisy mode)...
[2026-03-17 11:14:10]   Processed query 1/45: I am looking for Die-Namics dies... -> 10 results
[2026-03-17 11:14:10]   Processed query 1/45: I am look for Die-Namics die... -> 10 results
```

**问题**：虽然可以从查询文本推断（完整句子=clean，有错误=noisy），但这**不是显式的**，容易造成混淆。

## ✅ **实施的改进**

### **1. 修改evaluate_retriever()函数**

文件：`utils/utils.py` (第654行)

**改动**：
- 新增参数 `mode: str = None`
- 在日志中添加 `[clean]` 或 `[noisy]` 标记

```python
def evaluate_retriever(
    retriever, 
    queries: List[Dict], 
    all_asins: List[str], 
    k_values: List[int] = [1, 3, 5, 10], 
    save_candidates_path: str = None,
    mode: str = None  # ← 新增
) -> Dict:
    ...
    mode_tag = f"[{mode}]" if mode else ""
    log_with_timestamp(f"  Processed query {idx + 1}/{len(queries)} {mode_tag}: {query_text[:50]}... -> {len(results)} results")
```

### **2. 修改_evaluate_single_mode()调用**

文件：`12_evaluate_all_users_fullscale.py` (第176行)

**改动**：
- 传入 `mode=mode` 参数

```python
metrics = evaluate_retriever(retriever, queries, all_asins, k_values, mode=mode)
```

## 📊 **改进前后对比**

### **改进前**
```
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (clean mode)...
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (noisy mode)...
[2026-03-17 11:14:10]   Processed query 1/45: I am looking for Die-Namics dies... -> 10 results
[2026-03-17 11:14:10]   Processed query 1/45: I am look for Die-Namics die... -> 10 results
[2026-03-17 11:14:10]   Processed query 2/45: I am looking for Cottage Cutz... -> 10 results
[2026-03-17 11:14:10]   Processed query 2/45: I am look for Cottage Cutz... -> 10 results
```

❌ 需要推断查询文本质量来判断clean/noisy

### **改进后**
```
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (clean mode)...
[2026-03-17 11:14:10] Evaluating bm25 for A13OFOB1394G31 (noisy mode)...
[2026-03-17 11:14:10]   Processed query 1/45 [clean]: I am looking for Die-Namics dies... -> 10 results
[2026-03-17 11:14:10]   Processed query 1/45 [noisy]: I am look for Die-Namics die... -> 10 results
[2026-03-17 11:14:10]   Processed query 2/45 [clean]: I am looking for Cottage Cutz... -> 10 results
[2026-03-17 11:14:10]   Processed query 2/45 [noisy]: I am look for Cottage Cutz... -> 10 results
```

✅ 明确标记每一行是clean还是noisy

## 🔄 **向后兼容性**

- `mode` 参数有默认值 `None`
- 其他调用 `evaluate_retriever()` 的脚本无需修改
- 如果不传mode，日志中不会添加标记（保持原样）

## 📝 **受影响的脚本**

主要脚本：
- ✅ `12_evaluate_all_users_fullscale.py` - 已修改，现在传入mode

其他脚本（无需修改）：
- `12_evaluate_all_users.py`
- `12_reevaluate_from_cache.py`
- `12_reevaluate_incomplete_user.py`
- `specialized_evaluations/12_evaluate_retrieval.py`
- 各LLM reranking脚本

## 🎯 **预期效果**

### **可读性提升**
- 并发运行时，清晰看出每一行属于clean还是noisy
- 不需要对比查询文本内容来推断

### **调试便利性**
- 分析日志时更容易追踪问题
- 快速定位某个mode的性能问题

### **数据分析**
- 可以更精确地过滤和比较clean vs noisy的结果
- 日志解析工具更容易提取对应的metrics

## 📍 **验证方法**

运行改进后的评估脚本：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 ./.claude/skills/PersoanlQuery/12_retrieval/evaluators/12_evaluate_all_users_fullscale.py \
     --mode both --parallel 2 --users 1"
```

查看日志输出，应该看到：
```
  Processed query 1/45 [clean]: ...
  Processed query 1/45 [noisy]: ...
```

## 📌 **技术细节**

**为什么这个改进很重要**：

1. **并发导致的行交织** - Clean和Noisy的两个线程同时打印日志，行可能交织在一起
2. **查询文本相似** - 两个query文本内容几乎相同，难以区分
3. **日志解析困难** - 自动化脚本需要额外的逻辑来推断mode

这个改进通过**显式标记**解决了这个问题。
