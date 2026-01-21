# 统一颜色语义命名系统 - 设计建议

## 设计原则

### 1. **层次化结构**
- **基础色（Base）**：12种标准颜色
- **修饰符（Modifier）**：深浅、饱和度变化
- **特殊类型（Special）**：金属色、特殊效果

### 2. **命名规范**
- 全部小写
- 使用下划线分隔（`light_blue`）
- 避免缩写
- 语义清晰

### 3. **可扩展性**
- 预留 `unknown_*` 用于未分类颜色
- 支持复合颜色（如 `multicolor`）
- 支持特殊效果（如 `glow`, `pearl`）

---

## 完整颜色体系

### 基础颜色（14种）

| 标准名称 | 说明 | 常见变体 |
|---------|------|---------|
| `red` | 红色 | scarlet, carmine, wine, crimson |
| `orange` | 橙色 | burnt orange, vermillion |
| `yellow` | 黄色 | lemon, bright yellow, cadmium yellow |
| `green` | 绿色 | emerald, grass, olive, turquoise |
| `blue` | 蓝色 | cobalt, ultramarine, peacock, navy |
| `purple` | 紫色 | violet, lavender, lilac |
| `pink` | 粉色 | rose, flamingo, peach |
| `brown` | 棕色 | sepia, ochre, tan |
| `black` | 黑色 | - |
| `white` | 白色 | - |
| `gray` | 灰色 | grey, silver (非金属) |
| `beige` | 米色 | tan, cream |
| `aqua` | 青绿色 | teal, cyan |
| `clear` | 透明 | transparent |

### 修饰符系统

| 修饰符 | 格式 | 含义 | 示例 |
|-------|------|------|------|
| `light_` | `light_{color}` | 浅色/亮色 | `light_blue`, `light_pink` |
| `dark_` | `dark_{color}` | 深色 | `dark_green`, `dark_brown` |
| `vibrant_` | `vibrant_{color}` | 鲜艳/饱和 | `vibrant_red`, `vibrant_yellow` |
| `muted_` | `muted_{color}` | 柔和/低饱和 | `muted_green`, `muted_purple` |

**注意**：修饰符可以组合，但建议最多使用一个修饰符，保持简洁。

### 特殊类型

#### 金属色（Metallic）
| 类型 | 说明 | 映射来源 |
|------|------|---------|
| `metallic_gold` | 金色 | gold, antique gold, aztec gold, 14k |
| `metallic_silver` | 银色 | silver, antique silver, white gold |
| `metallic_copper` | 铜色 | copper, antique copper, sparkling copper |
| `metallic_bronze` | 青铜色 | bronze, antique bronze |
| `metallic_pewter` | 白镴色 | pewter |
| `metallic` | 通用金属色 | metallic (未指定具体金属) |

#### 特殊效果
| 类型 | 说明 | 映射来源 |
|------|------|---------|
| `pearl` | 珍珠效果 | pearl, macropearl, micropearl |
| `glow` | 发光效果 | glow in the dark, luminous, sparkle |
| `frost` | 霜冻效果 | frost |
| `multicolor` | 多色/混合色 | multicolored, assorted, variegated, any color |

---

## 命名规则详解

### 规则 1：基础色优先
- 优先使用基础色名称
- 例如：`Emerald Green` → `green`（而非 `emerald_green`）

### 规则 2：修饰符位置
- 修饰符在前，基础色在后
- 格式：`{modifier}_{base_color}`
- 例如：`light_blue`, `dark_red`

### 规则 3：特殊类型独立
- 特殊类型不使用修饰符
- 例如：`metallic_gold`（不是 `metallic_light_gold`）

### 规则 4：未分类颜色
- 使用 `unknown_{original_name}` 格式
- 例如：`Jewel Tones` → `unknown_jewel_tones`
- 后续可以手动分类或扩展规则

---

## 实施建议

### 阶段 1：基础映射（当前）
- ✅ 建立基础颜色体系
- ✅ 实现自动映射规则
- ✅ 生成映射表

### 阶段 2：优化映射规则
- 处理复合颜色（如 `Red/Blue` → `multicolor` 或 `red,blue`）
- 处理专业颜色名称（如 `Hooker's green` → `green`）
- 处理模糊描述（如 `Beautiful color` → 需要人工标注）

### 阶段 3：数据清洗
- 统一现有数据中的颜色命名
- 更新图构建脚本，使用标准化颜色
- 更新前端显示，使用标准化颜色

### 阶段 4：扩展与维护
- 建立颜色同义词词典
- 支持用户自定义颜色
- 定期审查 `unknown_*` 颜色，完善规则

---

## 使用示例

### 示例 1：基础颜色
```python
"Black" → "black"
"Blue" → "blue"
"Red" → "red"
```

### 示例 2：带修饰符
```python
"light Blue" → "light_blue"
"dark Brown" → "dark_brown"
"Bright Red" → "light_red"  # bright 视为 light
```

### 示例 3：金属色
```python
"Gold" → "metallic_gold"
"Antique Silver" → "metallic_silver"
"Copper" → "metallic_copper"
```

### 示例 4：特殊效果
```python
"Glow in the Dark" → "glow"
"Multicolored" → "multicolor"
"Pearl" → "pearl"
```

### 示例 5：未分类
```python
"Jewel Tones" → "unknown_jewel_tones"
"Water-soluble" → "unknown_water-soluble"  # 这不是颜色
```

---

## 统计信息

根据当前数据（102个产品，49个包含颜色）：
- **唯一原始颜色值**：180个
- **标准化后颜色数**：约 40-50个（取决于映射规则）
- **覆盖率**：预计 85-90%（剩余为 `unknown_*`）

---

## 后续优化方向

1. **颜色相似度计算**
   - 使用颜色空间（RGB/HSV）计算相似度
   - 自动合并相似颜色

2. **多语言支持**
   - 支持中文颜色名称
   - 支持其他语言的颜色描述

3. **颜色可视化**
   - 为每个标准颜色提供示例图片
   - 在 UI 中显示颜色预览

4. **用户反馈机制**
   - 允许用户纠正映射错误
   - 收集新的颜色变体

---

## 文件说明

- `color_schema_design.md` - 颜色体系设计文档
- `color_mapping.json` - 完整映射表（原始颜色 → 标准化颜色）
- `design_color_schema.py` - 映射脚本（可扩展）
