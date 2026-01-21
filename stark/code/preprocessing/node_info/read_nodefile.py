import pickle
import json

# 读取商品数据库
print("正在读取商品数据库...")
with open('/home/wlia0047/ar57/wenyu/stark/data/amazon/processed/node_info.pkl', 'rb') as f:
    node_info = pickle.load(f)

# 计算商品总数
if isinstance(node_info, dict):
    total_products = len(node_info)
    print(f"商品总数: {total_products}")
elif isinstance(node_info, list):
    total_products = len(node_info)
    print(f"商品总数: {total_products}")
else:
    print(f"数据类型: {type(node_info)}")
    try:
        total_products = len(node_info)
        print(f"商品总数: {total_products}")
    except:
        print("无法计算总数")

# 生成ID到ASIN的映射表
print("正在生成ID到ASIN的映射表...")
id_to_asin = {}

if isinstance(node_info, dict):
    for node_id, node_data in node_info.items():
        if isinstance(node_data, dict) and 'asin' in node_data:
            id_to_asin[str(node_id)] = node_data['asin']
        elif isinstance(node_data, list) and len(node_data) > 0:
            # 如果node_data是列表，假设第一个元素是ASIN
            id_to_asin[str(node_id)] = str(node_data[0])
elif isinstance(node_info, list):
    for i, node_data in enumerate(node_info):
        if isinstance(node_data, dict) and 'asin' in node_data:
            id_to_asin[str(i)] = node_data['asin']
        elif isinstance(node_data, list) and len(node_data) > 0:
            # 如果node_data是列表，假设第一个元素是ASIN
            id_to_asin[str(i)] = str(node_data[0])

# 保存映射表为JSON文件
output_path = '/home/wlia0047/ar57/wenyu/stark/data/amazon/processed/id_to_asin.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(id_to_asin, f, indent=2, ensure_ascii=False)

print(f"映射表已保存到: {output_path}")
print(f"映射表包含 {len(id_to_asin)} 个条目")

# 验证前几个映射
print("\n前10个映射示例:")
for i, (node_id, asin) in enumerate(list(id_to_asin.items())[:10]):
    print(f"  {node_id}: {asin}")
