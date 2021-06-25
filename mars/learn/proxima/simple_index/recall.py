# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import mars.remote as mr
from mars.learn.proxima.simple_index.knn import sample_data, linear_build_and_search


def recall_one(linear_score, ann_score, topk_ids, epsilon=1e-6):
    topk_matchs = {}
    for ids in topk_ids:
        topk_matchs[ids] = 0
    length = len(linear_score)
    match = 0
    idx, ann_item = 0, 0
    while idx < length:
        cur_topk = idx + 1
        if ann_item < len(ann_score):
            if math.fabs(linear_score[idx] - ann_score[ann_item]) < epsilon:
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
    topk_matchs, result_topk_matchs = {}, {}

    for ids in topk_ids:
        topk_matchs[ids] = 0
        result_topk_matchs[ids] = 0

    while idx < length:
        for k in topk_ids:
            dynamic_size = k
            while dynamic_size + 1 < length:
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
            result_topk_matchs[idx] = topk_matchs[idx] / idx

    return result_topk_matchs


def compute_recall(pk_l, distance_l, pk_p, distance_p, topk_ids, method="BYID", epsilon=1e-6):
    pk_l, distance_l, pk_p, distance_p = np.array(pk_l), np.array(distance_l), \
                                         np.array(pk_p), np.array(distance_p)
    topk_matchs = {}
    for ids in topk_ids:
        topk_matchs[ids] = 0
    for linear_res_k, linear_res_s, knn_res_k, knn_res_s in zip(pk_l, distance_l, pk_p, distance_p):
        if method == "BYID":
            res_t = recall_one_byid(linear_res_k, knn_res_k, knn_res_s, topk_ids)
        else:
            res_t = recall_one(linear_res_s, knn_res_s, topk_ids, epsilon)
        for k, v in res_t.items():
            topk_matchs[k] += v

    length = len(pk_l)
    for k, v in topk_matchs.items():
        topk_matchs[k] = min(v / length, 1)
    return topk_matchs


def recall(doc, query, topk, sample_count, pk_p, distance_p,
           row_number=None, column_number=None,
           topk_ids=None, method=None, epsilon=1e-6, session=None, run_kwargs=None):
    if topk_ids is None:
        topk_ids = [topk]
    if method is None:
        method = "BYSCORE"

    query_sample, idx = sample_data(query=query, sample_count=sample_count)
    pk_p_sample, distance_p_sample = pk_p[idx, :], distance_p[idx, :]
    pk_l, distance_l = linear_build_and_search(doc=doc, query=query_sample, topk=topk,
                                               row_number=row_number, column_number=column_number)

    r = mr.spawn(compute_recall, args=(pk_l, distance_l, pk_p_sample,
                                       distance_p_sample, topk_ids, method, epsilon))
    return r.execute(session=session, **(run_kwargs or dict())).fetch()
