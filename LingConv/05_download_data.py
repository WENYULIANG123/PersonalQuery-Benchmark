"""
Stage 5: 下载官方 LINGCONV 数据集
从 Hugging Face 加载 QQP、MRPC、STSB 并生成完整数据
"""
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


SOURCE_SPECS = {
    "qqp": {"sentence1": "question1", "sentence2": "question2", "keep": lambda label: label == 1},
    "mrpc": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label == 1},
    "stsb": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label >= 3},
}


def rename_dev_split(data):
    if "dev" not in data and "validation" in data:
        data["dev"] = data["validation"]
        del data["validation"]
    return data


def normalize_columns(data, source_name, sentence1_name, sentence2_name):
    for split in data.keys():
        split_data = data[split]
        rename_map = {}
        if sentence1_name in split_data.column_names and sentence1_name != "sentence1":
            rename_map[sentence1_name] = "sentence1"
        if sentence2_name in split_data.column_names and sentence2_name != "sentence2":
            rename_map[sentence2_name] = "sentence2"
        if rename_map:
            split_data = split_data.rename_columns(rename_map)
        if "source" not in split_data.column_names:
            split_data = split_data.add_column("source", [source_name] * len(split_data))
        keep_columns = [column for column in split_data.column_names if column.startswith("sentence") or column == "source"]
        data[split] = split_data.remove_columns(sorted(set(split_data.column_names) - set(keep_columns)))
    return data


def main():
    output_dir = "/home/wlia0047/ar57_scratch/wenyu/ling_conversion_official"
    os.makedirs(output_dir, exist_ok=True)

    datasets = {}
    for source_name in ["qqp", "mrpc", "stsb"]:
        print(f"加载 {source_name}...")
        spec = SOURCE_SPECS[source_name]
        dataset = load_dataset("glue", source_name)
        dataset = rename_dev_split(dataset)
        dataset = dataset.filter(lambda row: spec["keep"](row["label"]))
        dataset = normalize_columns(dataset, source_name, spec["sentence1"], spec["sentence2"])
        datasets[source_name] = dataset
        print(f"  {source_name}: train={len(dataset['train'])}, dev={len(dataset['dev'])}, test={len(dataset['test'])}")

    merged = DatasetDict(
        {
            "train": concatenate_datasets([dataset["train"] for dataset in datasets.values() if "train" in dataset]),
            "dev": concatenate_datasets([dataset["dev"] for dataset in datasets.values() if "dev" in dataset]),
            "test": concatenate_datasets([dataset["test"] for dataset in datasets.values() if "test" in dataset]),
        }
    )

    print(f"\n合并后: train={len(merged['train'])}, dev={len(merged['dev'])}, test={len(merged['test'])}")
    merged.save_to_disk(output_dir)
    print(f"数据已保存到: {output_dir}")


if __name__ == "__main__":
    main()
