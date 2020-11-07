import math
import numpy as np


def recall_one(linear_score, ann_score, topk_ids):
    topk_matchs = {}
    for ids in topk_ids:
        topk_matchs[ids] = 0
    length = len(linear_score)
    match = 0
    idx, ann_item = 0, 0
    while idx < length:
        cur_topk = idx + 1
        if ann_item < len(ann_score):
            if math.fabs(linear_score[idx] - ann_score[ann_item]) < 1e-6:
                ann_item += 1
                idx += 1
                match += 1
            else:
                if linear_score[idx] < ann_score[ann_item]:
                    idx += 1  # linear
                else:
                    ann_item += 1  # ann
        else:
            idx += 1

        if cur_topk in topk_ids:
            topk_matchs[cur_topk] = match / cur_topk

    return topk_matchs


def recall_one_byid(linear_key, ann_key, ann_score, topk_ids):
    idx, length = 0, len(linear_key)

    topk_matchs = {}
    for ids in topk_ids:
        topk_matchs[ids] = 0

    while idx < length:
        for k in topk_ids:
            dynamic_size = k
            while dynamic_size < length:
                if math.isclose(ann_score[dynamic_size - 1], ann_score[dynamic_size]):
                    dynamic_size += 1
                else:
                    break

            items = 0
            while items < len(ann_score) and items < dynamic_size:
                if linear_key[idx] == ann_key[items]:
                    topk_matchs[k] += 1
                    break
                else:
                    items += 1

        idx += 1
        if idx in topk_ids:
            topk_matchs[idx] = topk_matchs[idx] / idx

    return topk_matchs


def recall(pk_l, distance_l, pk_p, distance_p, topk_ids, method="BYID"):
    pk_l, distance_l, pk_p, distance_p = np.array(pk_l), np.array(distance_l), np.array(pk_p), np.array(distance_p)
    topk_matchs = {}
    for ids in topk_ids:
        topk_matchs[ids] = 0
    for linear_res_k, linear_res_s, knn_res_k, knn_res_s in zip(pk_l, distance_l, pk_p, distance_p):
        if method == "BYID":
            res_t = recall_one_byid(linear_res_k, knn_res_k, knn_res_s, topk_ids)
        else:
            res_t = recall_one(linear_res_s, knn_res_s, topk_ids)
        for k, v in res_t.items():
            topk_matchs[k] += v

    length = len(pk_l)
    for k, v in topk_matchs.items():
        topk_matchs[k] = v / length
    return topk_matchs
