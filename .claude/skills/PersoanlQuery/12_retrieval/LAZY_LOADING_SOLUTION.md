# 按需加载(Lazy Loading)解决方案

## 🎯 问题描述

日志中显示Dense Retrievers（E5、BGE、MiniLM、MPNet）在评估时连续失败：

```
11:36:51 Failed bge for A13OFOB1394G31
11:46:51 Failed e5 for A13OFOB1394G31  
11:56:51 Failed minilm for A13OFOB1394G31
12:06:51 Failed mpnet for A13OFOB1394G31
```

**根本原因**：
1. **并发设计缺陷** - Clean和Noisy模式同时评估，两个线程同时占用GPU
2. **内存架构不合理** - 所有302,380个embeddings一开始就加载到GPU上

## 🔧 解决方案

### 核心改进：BatchedLazyRetrieverWrapper

```python
class BatchedLazyRetrieverWrapper:
    """按需加载embeddings到GPU，批处理方式避免一次性加载302k"""
    
    def fit阶段():
        # ✅ Embeddings保存在CPU或磁盘
        # 不加载到GPU
        
    def search阶段():
        # 1. 分批加载embeddings到GPU (batch_size=5000)
        # 2. 计算这批的相似度
        # 3. 立即释放GPU内存
        # 4. 继续下一批
        # 5. 合并结果返回
```

### 内存占用对比

**原来的方式**：
```
GPU显存占用 = E5模型(1.3GB) + 302k embeddings(1.2GB) = 2.5GB持续占用
             + 如果有2个Dense模型同时运行 → 5GB+ ← OOM!
```

**新的方式**：
```
GPU显存占用 = E5模型(1.3GB) + 单批embeddings(5000个, ~24MB) = 1.4GB
             + 完全释放GPU缓存后继续
             + 多个Dense模型可以安全并发！
```

## 📝 实现文件

### 1. `lazy_retriever_wrapper.py` (新增)
- **LazyRetrieverWrapper** - 基础版本，一次性加载所有embeddings
- **BatchedLazyRetrieverWrapper** - 推荐版本，分批加载
  - 避免302k embeddings一次性加载
  - 自动GPU缓存管理
  - 支持多线程并发

### 2. `retriever_manager.py` (修改)
- 新增参数 `use_lazy_loading: bool = True`
- Dense Retrievers自动包装为BatchedLazyRetrieverWrapper
- Sparse Retrievers保持不变（CPU运算，无影响）

### 3. `test_lazy_loading.py` (新增)
- 测试脚本，验证GPU内存释放
- 对比直接加载 vs lazy加载的GPU占用

## 🚀 使用方法

### 自动启用（默认）

```python
from retriever_manager import get_retriever_manager

rm = get_retriever_manager()

# 自动应用lazy loading包装（Dense Retrievers）
e5 = rm.get_retriever('e5', documents, metadata)
bge = rm.get_retriever('bge', documents, metadata)

# 搜索时自动按需加载GPU
results = e5.search("query", top_k=10)
# GPU内存在搜索完后立即释放
```

### 禁用lazy loading（如需直接加载）

```python
e5 = rm.get_retriever('e5', documents, metadata, use_lazy_loading=False)
```

## ✅ 优势

1. **GPU内存效率** - 从2.5GB降至1.4GB
2. **并发安全** - Clean和Noisy可以安全并发
3. **自动批处理** - 无需用户手动管理batch_size
4. **向后兼容** - API不变，使用者无感知
5. **即插即用** - 无需修改评估脚本

## 📊 预期改进

| 指标 | 原来 | 改进后 |
|------|------|--------|
| GPU内存 | 2.5-3GB | 1.4GB |
| 同时运行Dense模型 | 1个 | 2-3个 |
| Clean+Noisy并发 | ❌ 超时 | ✅ 成功 |
| 搜索速度 | 快 | 快 (分批略慢) |

## 🔄 工作流程

```
用户调用: retriever.search("query")
    ↓
LazyRetrieverWrapper.search()
    ├─ 加载query embedding到GPU
    ├─ 逐批加载doc embeddings (batch_size=5000)
    │  └─ 计算这批的余弦相似度
    │  └─ 立即释放这批GPU内存
    │  └─ GPU缓存清理
    │  └─ 继续下一批
    ├─ 合并所有批次的结果
    └─ 返回top-k结果
```

## 📌 关键配置

```python
# lazy_retriever_wrapper.py 第200行
batch_size = 5000  # 每批5000个embeddings
# 调整此值可以平衡速度vs内存
# 更小 → 更省内存 (但搜索慢)
# 更大 → 更快 (但占用更多GPU)
```

## 🧪 测试

运行验证脚本：

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 ./.claude/skills/PersoanlQuery/12_retrieval/test_lazy_loading.py"
```

观察输出的GPU内存使用情况：
- 直接加载：应该看到2-3GB占用
- Lazy加载：应该看到1.4GB占用，搜索后立即释放

## 🎓 技术细节

### 为什么不直接修改E5Retriever？

因为需要保持向后兼容性。LazyRetrieverWrapper是一个**包装层**，可以：
- 与现有缓存系统兼容
- 随时启用/禁用
- 不影响其他代码

### 为什么选择批处理而不是逐个加载？

批处理的优势：
- **GPU吞吐量** - 批量计算cosine similarity更高效
- **内存对齐** - 避免频繁的内存碎片化
- **算力利用** - CUDA并行计算能充分发挥

单个加载的劣势：
- 计算效率低（302k次单个查询）
- 重复的CPU-GPU通信开销

## 🔮 未来改进

1. **FAISS集成** - 使用Facebook的FAISS库，进一步优化搜索速度
2. **动态batch_size** - 根据GPU内存自动调整batch_size
3. **索引压缩** - 使用Product Quantization减少内存占用
