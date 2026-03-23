#!/usr/bin/env python3
"""
数据集划分：
  - 原 pic/ 数据（前 N 条）→ val.jsonl（验证集）
  - 新采集 original/ 数据 → 按图片分组后随机划分 train_split.jsonl / test_split.jsonl

按图片分组划分（而非按样本随机），避免同一张图的不同指令样本泄露到两个集合。

用法：
    python split_dataset.py
    python split_dataset.py --test-ratio 0.15 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="merged_train.jsonl")
    parser.add_argument("--val-out", default="val.jsonl")
    parser.add_argument("--train-out", default="train_split.jsonl")
    parser.add_argument("--test-out", default="test_split.jsonl")
    parser.add_argument("--val-source", default="pic/",
                        help="images 路径前缀，匹配这个前缀的样本归入验证集")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="新数据中测试集比例（按图片数划分）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    # 分离验证集（原 pic/ 数据）和新数据
    val_samples = []
    new_samples = []
    for s in all_samples:
        img_path = s["images"][0] if s.get("images") else ""
        if img_path.startswith(args.val_source):
            val_samples.append(s)
        else:
            new_samples.append(s)

    # 按图片分组新数据
    img_to_samples: dict[str, list] = defaultdict(list)
    for s in new_samples:
        img_to_samples[s["images"][0]].append(s)

    images = sorted(img_to_samples.keys())
    random.shuffle(images)

    n_test_imgs = max(1, round(len(images) * args.test_ratio))
    test_images = set(images[:n_test_imgs])
    train_images = set(images[n_test_imgs:])

    train_samples = [s for img in train_images for s in img_to_samples[img]]
    test_samples  = [s for img in test_images  for s in img_to_samples[img]]

    # 写出
    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(args.val_out,   val_samples)
    write_jsonl(args.train_out, train_samples)
    write_jsonl(args.test_out,  test_samples)

    print(f"验证集  {args.val_out:25s}: {len(val_samples):4d} 条  ({len(set(s['images'][0] for s in val_samples))} 张图)")
    print(f"训练集  {args.train_out:25s}: {len(train_samples):4d} 条  ({len(train_images)} 张图)")
    print(f"测试集  {args.test_out:25s}: {len(test_samples):4d} 条  ({len(test_images)} 张图)")


if __name__ == "__main__":
    main()
