from collections import defaultdict
import numpy as np

def calculate_ap(recall, precision):
    """根据召回率和精度计算平均精度(AP)"""
    # 首先对精度进行排序
    recall, precision = zip(*sorted(zip(recall, precision), key=lambda x: x[0], reverse=True))
    # 初始化AP
    ap = 0.0
    # 计算AP
    for t in np.arange(0.0, 1.01, 0.01):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap

def calculate_map(detections, ground_truths, iou_threshold=0.5, class_agnostic=False):
    """
    计算mAP
    :param detections: 检测结果列表，每个元素是一个字典，包含'image_id', 'category_id', 'bbox', 'score' 
    :param ground_truths: 真实标签列表，每个元素是一个字典，包含'image_id', 'category_id', 'bbox'
    :param iou_threshold: 交并比阈值
    :param class_agnostic: 是否忽略类别
    :return: mAP值
    """
    # 初始化
    ap_sum = 0.0
    categories = set()
    for detection in detections:
        categories.add(detection['category_id'])

    # 对每个类别计算AP
    for category in sorted(categories):
        if class_agnostic:
            relevant_detections = [d for d in detections if d['score'] > 0]
        else:
            relevant_detections = [d for d in detections if d['category_id'] == category]
        relevant_ground_truths = [gt for gt in ground_truths if gt['category_id'] == category]

        # 计算每个检测框与真实框的IoU
        ious = np.zeros((len(relevant_detections), len(relevant_ground_truths)))
        for i, det in enumerate(relevant_detections):
            for j, gt in enumerate(relevant_ground_truths):
                ious[i, j] = calculate_iou(det['bbox'], gt['bbox'])

        # 计算AP
        ap = calculate_ap_for_category(relevant_detections, relevant_ground_truths, ious, iou_threshold)
        ap_sum += ap

    # 计算mAP
    map_value = ap_sum / len(categories)
    return map_value

def calculate_iou(boxA, boxB):
    """计算两个边界框的交并比"""
    # 获取边界框的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集面积
    unionArea = boxAArea + boxBArea - interArea

    # 计算交并比
    iou = interArea / float(unionArea)

    return iou

def calculate_ap_for_category(detections, ground_truths, ious, iou_threshold):
    """针对一个类别计算AP"""
    true_positives = np.zeros(len(detections))
    false_positives = np.zeros(len(detections))
    detections_matched = np.zeros(len(ground_truths))

    # 对每个检测框进行评估
    for d, det in enumerate(detections):
        # 找到与当前检测框IoU最高的真实框
        best_gt_index = np.argmax(ious[d])
        best_iou = ious[d][best_gt_index]

        # 如果IoU高于阈值，则认为是正样本
        if best_iou > iou_threshold and detections_matched[best_gt_index] == 0:
            true_positives[d] = 1
            detections_matched[best_gt_index] = 1
        else:
            false_positives[d] = 1

    # 计算累积的真正例和假正例
    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)

    # 计算召回率和精度
    recalls = cum_true_positives / len(ground_truths)
    precisions = cum_true_positives / (cum_true_positives + cum_false_positives)

    # 计算AP
    ap = calculate_ap(recalls, precisions)
    return ap

# 示例使用
# 假设detections和ground_truths已经定义好了
# detections = [{'image_id': 1, 'category_id': 1, 'bbox': [x1, y1, x2, y2], 'score': score1}, ...]
# ground_truths = [{'image_id': 1, 'category_id': 1, 'bbox': [x1, y1, x2, y2]}, ...]
# map_value = calculate_map(detections, ground_truths)

if __name__ == '__main__':
    calculate_map()