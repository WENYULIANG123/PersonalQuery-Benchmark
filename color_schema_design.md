# 统一颜色语义命名系统

## 1. 基础颜色（Base Colors）

| 标准名称 | 别名/变体 |
|---------|----------|
| `red` | red, carmine, scarlet, carnival, wine... |
| `orange` | orange, burnt orange, vermillion |
| `yellow` | yellow, lemon, bright yellow, cadmium yellow, hansa yellow |
| `green` | green, emerald, grass, olive, hooker... |
| `blue` | blue, cobalt, ultramarine, french ultramarine, peacock... |
| `purple` | purple, violet, carbazole violet, deep violet, light violet... |
| `pink` | pink, flamingo, peach, pale, rose... |
| `brown` | brown, mid brown, dark brown, light brown, deep brown... |
| `black` | black, gloss black, polished black, black coarse, black fine |
| `white` | white, pearlwhite, white coarse, white fine |
| `gray` | gray, grey, blue gray, green gray, mid gray... |
| `beige` | beige, brick beige |
| `aqua` | aqua, teal, turquoise |
| `clear` | clear |

## 2. 修饰符（Modifiers）

| 修饰符 | 含义 | 关键词 |
|-------|------|--------|
| `light_` | 浅色/亮色 | light, pale, bright |
| `dark_` | 深色 | dark, deep |
| `vibrant_` | 鲜艳 | vibrant, brilliant, gorgeous |
| `muted_` | 柔和 | muted, dull, shadow |

## 3. 特殊类型（Special Types）

| 类型 | 说明 | 示例 |
|------|------|------|
| `metallic_gold` | 金色金属 | gold, antique gold, aztec gold |
| `metallic_silver` | 银色金属 | silver, antique silver |
| `metallic_copper` | 铜色金属 | copper, antique copper, sparkling copper |
| `metallic_bronze` | 青铜色 | bronze, antique bronze |
| `metallic_pewter` | 白镴色 | pewter |
| `metallic` | 通用金属色 | metallic |
| `pearl` | 珍珠效果 | pearl, macropearl, micropearl |
| `glow` | 发光效果 | glow in the dark, luminous, sparkle |
| `frost` | 霜冻效果 | frost |
| `multicolor` | 多色 | multicolored, assorted, variegated |

## 4. 命名规则

### 格式：`[modifier_]base_color` 或 `special_type`

### 示例：
- `red` - 标准红色
- `light_blue` - 浅蓝色
- `dark_green` - 深绿色
- `vibrant_yellow` - 鲜艳黄色
- `metallic_gold` - 金色金属
- `pearl` - 珍珠效果
- `multicolor` - 多色

## 5. 映射示例

| 原始颜色 | 标准化后 |
|---------|---------|
| `Black` | `black` |
| `black` | `black` |
| `Blue` | `blue` |
| `light Blue` | `light_blue` |
| `dark Brown` | `dark_brown` |
| `Gold` | `metallic_gold` |
| `Antique Gold` | `metallic_gold` |
| `Metallic` | `metallic` |
| `Red` | `red` |
| `Bright Red` | `light_red` |
| `Violet` | `purple` |
| `deep Violet` | `dark_purple` |
| `Green` | `green` |
| `Emerald Green` | `green` |
| `Pink` | `pink` |
| `Flamingo Pink` | `pink` |
| `White` | `white` |
| `Pearlwhite` | `pearl` |
| `Multicolored` | `multicolor` |
| `Assorted Color` | `multicolor` |
| `Glow in the Dark` | `glow` |