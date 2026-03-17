# HuggingFace语法错误检测模型

## 推荐模型 ⭐

### sahilnishad/BERT-GED-FCE-FT
- **类型**: Token Classification (Binary - Correct/Incorrect)
- **基础**: BERT-base-uncased (110M parameters)
- **数据集**: FCE (First Certificate in English)
- **用途**: 英文语法错误检测
- **优点**: 
  - ✓ 生产就绪，直接支持AutoModelForTokenClassification
  - ✓ 基于标准BERT，无需自定义代码
  - ✓ 在FCE标准基准上微调
  - ✓ 二元分类：正确/不正确标记
  - ✓ 适合产品评论等文本的错误检测

### 如何使用

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_id = "sahilnishad/BERT-GED-FCE-FT"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(model_id)
```

### 预期输出格式

模型输出二元分类结果：
- `0`: CORRECT - 标记语法正确
- `1`: INCORRECT - 标记有语法错误

模型以概率形式输出，您可以设置阈值来调整检测敏感度。

### 标签类型

- **Binary Classification**:
  - Class 0: `CORRECT` (grammatically correct)
  - Class 1: `INCORRECT` (grammatically incorrect)
- **Error Type**: 统一分类为 `GRAMMAR` (模型不区分具体错误类型)

## 性能指标

**BERT-based GED (语法错误检测)** 在FCE数据集上的典型性能:
- **F1 Score**: 80-90% (state-of-the-art for BERT-based models)
- 超越传统规则系统 (F1 50-60%)
- 接近最先进的DeBERTa模型

## 使用示例

```bash
python batch_detect_hf.py input.json output.json "sahilnishad/BERT-GED-FCE-FT"
```

或使用默认模型:
```bash
python batch_detect_hf.py input.json output.json
```

## 输入格式

```json
[
  "The quality are excellent.",
  "I has received the item.",
  "It work great."
]
```

## 输出格式

```json
[
  {
    "text": "The quality are excellent.",
    "errors": [
      {
        "token_idx": 2,
        "token": "are",
        "prediction": "INCORRECT",
        "incorrect_probability": 0.89,
        "correct_probability": 0.11,
        "error_type": "GRAMMAR"
      }
    ],
    "error_count": 1
  },
  {
    "text": "I has received the item.",
    "errors": [
      {
        "token_idx": 1,
        "token": "has",
        "prediction": "INCORRECT",
        "incorrect_probability": 0.95,
        "correct_probability": 0.05,
        "error_type": "GRAMMAR"
      }
    ],
    "error_count": 1
  }
]
```
