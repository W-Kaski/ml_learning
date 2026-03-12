#!/usr/bin/env python3
"""
CNN 实战 - 目标检测入门（YOLO 前置基础）
=====================================

学习目标：
1. 理解目标检测中的边界框（bbox）与置信度（score）
2. 掌握 NMS（非极大值抑制）去重逻辑
3. 完成一个最小可运行检测后处理示例

说明：
- 这是“检测后处理”教学脚本，不依赖 YOLO 权重下载。
- 重点是先把检测任务的核心数据结构理解清楚。
"""

import torch
from torchvision.ops import nms, box_iou

print("=" * 70)
print("🎯 目标检测基础：边界框 + NMS")
print("=" * 70)


# ============================================================================
# 1. 构造候选框与分类分数
# ============================================================================
print("\n【1. 构造候选检测框】")

# 边界框格式: [x1, y1, x2, y2]
boxes = torch.tensor([
    [10, 10, 50, 50],   # 框A
    [12, 12, 48, 48],   # 框B（与A高度重叠）
    [100, 100, 150, 150],  # 框C（远离A/B）
    [105, 105, 152, 152],  # 框D（与C重叠）
    [200, 30, 240, 80],    # 框E（独立）
], dtype=torch.float32)

scores = torch.tensor([0.95, 0.80, 0.88, 0.70, 0.60], dtype=torch.float32)

print(f"候选框数量: {len(boxes)}")
for i in range(len(boxes)):
    print(f"  框{i}: {boxes[i].tolist()}, score={scores[i].item():.2f}")


# ============================================================================
# 2. 计算 IoU（重叠度）
# ============================================================================
print("\n【2. IoU 重叠分析】")

iou_mat = box_iou(boxes, boxes)
print("IoU 矩阵（保留两位小数）:")
print(torch.round(iou_mat * 100) / 100)

print("\n解释：")
print("- A 与 B 的 IoU 高，说明重复检测概率大")
print("- C 与 D 的 IoU 也较高")
print("- E 与其他框 IoU 低，通常会被保留")


# ============================================================================
# 3. 执行 NMS 去重
# ============================================================================
print("\n【3. 执行 NMS】")

iou_threshold = 0.5
keep_indices = nms(boxes, scores, iou_threshold=iou_threshold)

print(f"NMS IoU 阈值: {iou_threshold}")
print(f"保留索引: {keep_indices.tolist()}")

kept_boxes = boxes[keep_indices]
kept_scores = scores[keep_indices]

print("\nNMS 后保留框:")
for rank, idx in enumerate(keep_indices.tolist()):
    print(f"  Top{rank+1}: 原框{idx}, bbox={boxes[idx].tolist()}, score={scores[idx].item():.2f}")


# ============================================================================
# 4. 模拟“分类 + 回归”后处理输出
# ============================================================================
print("\n【4. 模拟检测结果输出】")

class_names = ["person", "car", "dog", "cat", "bottle"]
pred_labels = torch.tensor([0, 0, 1, 1, 4])  # 假设前两个是 person，中间两个是 car

print("NMS 前预测类别:")
for i in range(len(boxes)):
    print(f"  框{i}: class={class_names[pred_labels[i]]}, score={scores[i].item():.2f}")

print("\nNMS 后最终检测结果:")
for idx in keep_indices.tolist():
    cls = class_names[pred_labels[idx]]
    bbox = boxes[idx].tolist()
    score = scores[idx].item()
    print(f"  class={cls:7s} score={score:.2f} bbox={bbox}")


# ============================================================================
# 5. 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ object_detection.py 运行完成")
print("=" * 70)
print("\n💡 关键要点：")
print("  1) 检测模型会输出很多候选框，必须做 NMS 去重")
print("  2) IoU 阈值越小，抑制越强；阈值越大，保留越多")
print("  3) 实战中常按类别分别做 NMS（class-wise NMS）")
print("=" * 70)
