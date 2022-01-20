#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import argparse
import time

import mars
import mars.dataframe as md
import pandas as pd


def load_lineitem(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/lineitem.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_part(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/part.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_orders(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/orders.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_customer(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/customer.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_nation(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/nation.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_region(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/region.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_supplier(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/supplier.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def load_partsupp(data_folder: str) -> md.DataFrame:
    data_path = data_folder + "/partsupp.pq"
    df = md.read_parquet(
        data_path,
    )
    return df


def q01(lineitem: md.DataFrame):
    t1 = time.time()
    date = pd.Timestamp("1998-09-02")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "L_DISCOUNT",
            "L_TAX",
            "L_RETURNFLAG",
            "L_LINESTATUS",
            "L_SHIPDATE",
            "L_ORDERKEY",
        ],
    ]
    sel = lineitem_filtered.L_SHIPDATE <= date
    lineitem_filtered = lineitem_filtered[sel]
    lineitem_filtered["AVG_QTY"] = lineitem_filtered.L_QUANTITY
    lineitem_filtered["AVG_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE
    lineitem_filtered["DISC_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE * (
        1 - lineitem_filtered.L_DISCOUNT
    )
    lineitem_filtered["CHARGE"] = (
        lineitem_filtered.L_EXTENDEDPRICE
        * (1 - lineitem_filtered.L_DISCOUNT)
        * (1 + lineitem_filtered.L_TAX)
    )
    gb = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)[
        "L_QUANTITY",
        "L_EXTENDEDPRICE",
        "DISC_PRICE",
        "CHARGE",
        "AVG_QTY",
        "AVG_PRICE",
        "L_DISCOUNT",
        "L_ORDERKEY",
    ]
    total = gb.agg(
        {
            "L_QUANTITY": "sum",
            "L_EXTENDEDPRICE": "sum",
            "DISC_PRICE": "sum",
            "CHARGE": "sum",
            "AVG_QTY": "mean",
            "AVG_PRICE": "mean",
            "L_DISCOUNT": "mean",
            "L_ORDERKEY": "count",
        }
    )
    total = total.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    print(total.execute())
    print("Q01 Execution time (s): ", time.time() - t1)


def q02(part, partsupp, supplier, nation, region):
    t1 = time.time()
    nation_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME", "N_REGIONKEY"]]
    region_filtered = region[(region["R_NAME"] == "EUROPE")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    r_n_merged = nation_filtered.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    r_n_merged = r_n_merged.loc[:, ["N_NATIONKEY", "N_NAME"]]
    supplier_filtered = supplier.loc[
        :,
        [
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_NATIONKEY",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    s_r_n_merged = r_n_merged.merge(
        supplier_filtered, left_on="N_NATIONKEY", right_on="S_NATIONKEY", how="inner"
    )
    s_r_n_merged = s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY", "PS_SUPPLYCOST"]]
    ps_s_r_n_merged = s_r_n_merged.merge(
        partsupp_filtered, left_on="S_SUPPKEY", right_on="PS_SUPPKEY", how="inner"
    )
    ps_s_r_n_merged = ps_s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_PARTKEY",
            "PS_SUPPLYCOST",
        ],
    ]
    part_filtered = part.loc[:, ["P_PARTKEY", "P_MFGR", "P_SIZE", "P_TYPE"]]
    part_filtered = part_filtered[
        (part_filtered["P_SIZE"] == 15)
        & (part_filtered["P_TYPE"].str.endswith("BRASS"))
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_MFGR"]]
    merged_df = part_filtered.merge(
        ps_s_r_n_merged, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    merged_df = merged_df.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_SUPPLYCOST",
            "P_PARTKEY",
            "P_MFGR",
        ],
    ]
    min_values = merged_df.groupby("P_PARTKEY", as_index=False)["PS_SUPPLYCOST"].min()
    min_values.columns = ["P_PARTKEY_CPY", "MIN_SUPPLYCOST"]
    merged_df = merged_df.merge(
        min_values,
        left_on=["P_PARTKEY", "PS_SUPPLYCOST"],
        right_on=["P_PARTKEY_CPY", "MIN_SUPPLYCOST"],
        how="inner",
    )
    total = merged_df.loc[
        :,
        [
            "S_ACCTBAL",
            "S_NAME",
            "N_NAME",
            "P_PARTKEY",
            "P_MFGR",
            "S_ADDRESS",
            "S_PHONE",
            "S_COMMENT",
        ],
    ]
    total = total.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    print(total.execute())
    print("Q02 Execution time (s): ", time.time() - t1)


def run_queries(data_folder: str):
    mars.new_session()

    # Load the data
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)
    supplier = load_supplier(data_folder)
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    mars.execute([lineitem, orders, customer, nation, region, supplier, part, partsupp])
    print("Reading time (s): ", time.time() - t1)

    q01(lineitem)
    q02(part, partsupp, supplier, nation, region)


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        help="The folder containing TPCH data",
    )
    args = parser.parse_args()
    folder = args.folder
    run_queries(folder)


if __name__ == "__main__":
    main()
