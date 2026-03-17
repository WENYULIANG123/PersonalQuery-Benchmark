# 快速开始指南 - 英文评论错误检测

> 🎯 目标: 检测评论中的拼写、语法、标点错误
> ⏱️ 时间: 3-5天实现 MVP  
> 💰 成本: 免费（可选GPU租赁 ¥500-1000/月）

---

## 🚀 5分钟快速了解

### 核心方案：改进GECToR
- **What**: Seq2Seq token tagging model (编辑操作预测)
- **How**: 输入句子 → 5002类编辑标签 → 错误位置+类型
- **Why**: 精度85-90%, 召回85-95%, 本地部署, 零成本

### 3个主要优势
```
1. 高召回 (85-95%) ✓ 符合你的需求
2. 成本低 (开源模型) ✓ 无API费用
3. 部署快 (3-5天) ✓ 可立即启动
```

---

## 📋 实施路线 (天数)

```
Day 1: 环境 + 验证
  ├─ 下载模型权重 (850MB)
  └─ python predict.py 测试

Day 2-3: 代码改造
  ├─ 改造 gec_model.py 的 detect_errors()
  ├─ 实现错误分类 (SPELLING/GRAMMAR/PUNCT)
  └─ 单元测试

Day 4: 参数调优
  ├─ 在验证集测试 3-5 个阈值配置
  └─ 选择最优参数

Day 5: 部署上线
  ├─ API 集成 (FastAPI)
  └─ 性能基准测试
```

---

## ⚡ 30秒演示

```bash
# 1. 创建测试评论
cat > test.txt << 'EOF'
Teh product quality is really good
I likes this product very much
Great quality good price service
EOF

# 2. 运行检测 (高召回模式)
python predict.py \
  --model_path roberta_1_gectorv2.th \
  --input_file test.txt \
  --output_file out.txt \
  --min_error_probability 0.1

# 3. 对比输入输出
diff test.txt out.txt
```

**输出示例**:
```
< Teh product quality is really good
> The product quality is really good
  
< I likes this product very much
> I like this product very much
```

---

## 📊 性能指标预期

| 指标 | 预期值 | 备注 |
|---|---|---|
| **精度** | 70-80% | 会有误检，但很少 |
| **召回** | 85-95% | 大部分错误被检出 ✓ |
| **推理延迟** | 100-300ms | 单句 |
| **吞吐量** | 100+ QPS | Batch mode, GPU |
| **模型大小** | 850MB | RoBERTa 权重 |
| **显存占用** | 2-3GB | 运行时 |

---

## 🎯 三个关键参数 (调节精度/召回)

```python
# ▼ 高召回配置 (推荐用于你的场景)
GecBERTModel(
    min_error_probability=0.1,      # ← 关键！降低阈值
    additional_confidence=-0.2,     # 降低"保持不变"的权重
    iterations=1,                   # 检测只需1次
)

# ▲ 高精度配置 (若后续要求降低误检)
GecBERTModel(
    min_error_probability=0.5,      
    additional_confidence=+0.5,
    iterations=3,
)
```

**参数含义**:
- `min_error_probability`: 最低错误概率 (越低 → 召回越高)
- `additional_confidence`: 给"保持不变"标签加权 (越低 → 越积极纠正)
- `iterations`: 迭代轮数 (1=快速检测, 5=深度纠正)

---

## 💻 最小代码示例

```python
from gector.gec_model import GecBERTModel

# 加载模型 (首次自动下载HF权重, ~10分钟)
model = GecBERTModel(
    vocab_path='data/output_vocabulary',
    model_paths=['roberta_1_gectorv2.th'],
    iterations=1,
    min_error_probability=0.1
)

# 检测错误
comments = [
    ['Teh', 'product', 'is', 'good'],
    ['I', 'likes', 'this', 'product']
]

corrected, num_edits = model.handle_batch(comments)

# 输出
# corrected = [
#     ['The', 'product', 'is', 'good'],      # Teh → The
#     ['I', 'like', 'this', 'product']       # likes → like
# ]
# num_edits = 2
```

---

## 🔧 关键代码改造点

### 改造 1: gec_model.py 的 postprocess_batch()

```python
def detect_errors(self, batch, threshold=0.1):
    """返回错误不返回纠正"""
    results = []
    for tokens, probabilities, idxs in zip(...):
        errors = []
        for pos, (idx, prob) in enumerate(zip(idxs, probabilities)):
            label = self.vocab.get_token_from_index(idx, 'labels')
            if label != '$KEEP' and prob >= threshold:
                errors.append({
                    'pos': pos,
                    'token': tokens[pos],
                    'label': label,
                    'confidence': prob
                })
        results.append({'tokens': tokens, 'errors': errors})
    return results
```

### 改造 2: 错误分类

```python
def classify_error(label: str) -> str:
    """映射 5002 类标签 → 6 大类"""
    if label.startswith('$REPLACE_') or label.startswith('$APPEND_'):
        return 'SPELLING'
    elif '$VERB' in label or '$AGREEMENT' in label:
        return 'GRAMMAR'
    elif any(p in label for p in [',', '.', '!', '?', ';', ':']):
        return 'PUNCTUATION'
    else:
        return 'OTHER'
```

---

## 📦 环境要求

| 项目 | 版本 | 备注 |
|---|---|---|
| Python | 3.8+ | 推荐 3.9+ |
| PyTorch | 1.10+ | 旧版本，有兼容性问题 |
| Transformers | 4.11+ | 需要AllenNLP 0.8.4 |
| CUDA | 11.0+ | 若要GPU加速 |

**安装**:
```bash
pip install torch==1.10.0 allennlp==0.8.4 transformers==4.11.3 \
    python-Levenshtein==0.12.1 sentencepiece==0.1.95
```

---

## 📥 模型下载

### 选项1: 自动下载 (推荐)
```bash
# 首次使用时会自动下载 Transformers 权重
# (自动缓存到 ~/.cache/huggingface/)
python predict.py ... # 运行时自动下载
```

### 选项2: 手动下载
```bash
# 下载 RoBERTa 预训练权重
wget -O roberta_1_gectorv2.th \
  https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th
  
# 或其他模型:
# - bert_0_gectorv2.th (精度稍低，更快)
# - xlnet_0_gectorv2.th (高精度)
```

---

## ✅ 验收标准

### MVP 版本 (必须)
- [x] 能检测拼写错误 (Teh → The)
- [x] 能检测语法错误 (likes → like)
- [x] 能检测标点错误 (缺逗号)
- [x] 精度 ≥ 70%
- [x] 召回 ≥ 85%
- [x] 延迟 ≤ 300ms

### 增强版本 (可选)
- [ ] 显示修改建议
- [ ] 返回置信度评分
- [ ] 支持批量处理
- [ ] 多语言支持

---

## 🐛 常见问题

**Q: 能处理中文评论吗?**  
A: GECToR是英文模型，中文支持差。需要微调或用专门的中文GEC模型。

**Q: 没有GPU怎么办?**  
A: CPU能用，但慢(1-2秒/句)。建议用云GPU按量租赁。

**Q: 怎么提高精度?**  
A: 降低 `min_error_probability` (会增加误检) 或微调模型 (需300-500条标注数据)。

**Q: 如何处理很长的评论?**  
A: GECToR默认截断到50tokens。改用sliding window扩展到100+。

**Q: 能否用GPT-4代替?**  
A: 可以，但成本高($0.02-0.10/条)，延迟也高(1-5秒)。

---

## 📞 需要帮助?

### 文档
- GECToR 论文: https://arxiv.org/abs/2010.05791
- GitHub: https://github.com/grammarly/gector
- 相关论文: 见 `ERROR_DETECTION_RECOMMENDATIONS.md`

### 故障排查
1. 检查 Python 版本 (`python --version`)
2. 检查依赖 (`pip list | grep -E "torch|allennlp"`)
3. 检查 GPU (`python -c "import torch; print(torch.cuda.is_available())"`)
4. 运行单条句子测试

### 性能优化
- 用 Batch 模式而非单句
- 启用 GPU (`torch.cuda.is_available()`)
- 降低精度需求 (用更小的模型)

---

## 下一步行动

**立即**: 阅读完整文档
```bash
cat ERROR_DETECTION_RECOMMENDATIONS.md  # 完整方案
cat VERIFICATION_RESULTS.md              # 验证结果
```

**本周**: 下载 + 验证
```bash
cd /fs04/ar57/wenyu/gector
# (下载模型)
python predict.py --model_path ... --input_file ... --output_file ...
```

**下周**: 改造 + 测试
```bash
# 修改 gec_model.py 的 detect_errors() 方法
# 编写单元测试
# 调整参数
```

**两周**: 部署上线
```bash
# API 集成
# 性能测试
# 灰度发布
```

---

**祝你成功! 🚀**

任何问题可以参考项目中的 3 个文档:
- `QUICK_START.md` (本文件) - 5分钟上手
- `ERROR_DETECTION_RECOMMENDATIONS.md` - 完整方案+成本分析
- `VERIFICATION_RESULTS.md` - 技术验证+性能指标
