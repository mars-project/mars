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


import numpy as np
import pandas as pd


def load_lineitem(fpath):
    cols_names = [
        "L_ORDERKEY",
        "L_PARTKEY",
        "L_SUPPKEY",
        "L_LINENUMBER",
        "L_QUANTITY",
        "L_EXTENDEDPRICE",
        "L_DISCOUNT",
        "L_TAX",
        "L_RETURNFLAG",
        "L_LINESTATUS",
        "L_SHIPDATE",
        "L_COMMITDATE",
        "L_RECEIPTDATE",
        "L_SHIPINSTRUCT",
        "L_SHIPMODE",
        "L_COMMENT",
    ]
    cols = {
        "L_ORDERKEY": np.int64,
        "L_PARTKEY": np.int64,
        "L_SUPPKEY": np.int64,
        "L_LINENUMBER": np.int64,
        "L_QUANTITY": np.float64,
        "L_EXTENDEDPRICE": np.float64,
        "L_DISCOUNT": np.float64,
        "L_TAX": np.float64,
        "L_RETURNFLAG": str,
        "L_LINESTATUS": str,
        "L_SHIPDATE": str,
        "L_COMMITDATE": str,
        "L_RECEIPTDATE": str,
        "L_SHIPINSTRUCT": str,
        "L_SHIPMODE": str,
        "L_COMMENT": str,
    }
    rel = pd.read_csv(
        fpath,
        sep="|",
        header=None,
        names=cols_names,
        dtype=cols,
        parse_dates=[10, 11, 12],
    )
    return rel


def load_lineitem_with_date(fpath):
    cols_names = [
        "L_ORDERKEY",
        "L_PARTKEY",
        "L_SUPPKEY",
        "L_LINENUMBER",
        "L_QUANTITY",
        "L_EXTENDEDPRICE",
        "L_DISCOUNT",
        "L_TAX",
        "L_RETURNFLAG",
        "L_LINESTATUS",
        "L_SHIPDATE",
        "L_COMMITDATE",
        "L_RECEIPTDATE",
        "L_SHIPINSTRUCT",
        "L_SHIPMODE",
        "L_COMMENT",
    ]
    cols = {
        "L_ORDERKEY": np.int64,
        "L_PARTKEY": np.int64,
        "L_SUPPKEY": np.int64,
        "L_LINENUMBER": np.int64,
        "L_QUANTITY": np.float64,
        "L_EXTENDEDPRICE": np.float64,
        "L_DISCOUNT": np.float64,
        "L_TAX": np.float64,
        "L_RETURNFLAG": str,
        "L_LINESTATUS": str,
        "L_SHIPDATE": str,
        "L_COMMITDATE": str,
        "L_RECEIPTDATE": str,
        "L_SHIPINSTRUCT": str,
        "L_SHIPMODE": str,
        "L_COMMENT": str,
    }
    rel = pd.read_csv(
        fpath,
        sep="|",
        header=None,
        names=cols_names,
        dtype=cols,
        parse_dates=[10, 11, 12],
    )
    rel["L_SHIPDATE"] = [time.date() for time in rel["L_SHIPDATE"]]
    rel["L_COMMITDATE"] = [time.date() for time in rel["L_COMMITDATE"]]
    rel["L_RECEIPTDATE"] = [time.date() for time in rel["L_RECEIPTDATE"]]
    return rel


def load_part(fpath):
    cols_names = [
        "P_PARTKEY",
        "P_NAME",
        "P_MFGR",
        "P_BRAND",
        "P_TYPE",
        "P_SIZE",
        "P_CONTAINER",
        "P_RETAILPRICE",
        "P_COMMENT",
    ]
    cols = {
        "P_PARTKEY": np.int64,
        "P_NAME": str,
        "P_MFGR": str,
        "P_BRAND": str,
        "P_TYPE": str,
        "P_SIZE": np.int64,
        "P_CONTAINER": str,
        "P_RETAILPRICE": np.float64,
        "P_COMMENT": str,
    }
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel


def load_orders(fpath):
    cols_names = [
        "O_ORDERKEY",
        "O_CUSTKEY",
        "O_ORDERSTATUS",
        "O_TOTALPRICE",
        "O_ORDERDATE",
        "O_ORDERPRIORITY",
        "O_CLERK",
        "O_SHIPPRIORITY",
        "O_COMMENT",
    ]
    cols = {
        "O_ORDERKEY": np.int64,
        "O_CUSTKEY": np.int64,
        "O_ORDERSTATUS": str,
        "O_TOTALPRICE": np.float64,
        "O_ORDERDATE": np.int64,
        "O_ORDERPRIORITY": str,
        "O_CLERK": str,
        "O_SHIPPRIORITY": np.int64,
        "O_COMMENT": str,
    }
    rel = pd.read_csv(
        fpath, sep="|", header=None, names=cols_names, dtype=cols, parse_dates=[4]
    )
    return rel


def load_orders_with_date(fpath):
    cols_names = [
        "O_ORDERKEY",
        "O_CUSTKEY",
        "O_ORDERSTATUS",
        "O_TOTALPRICE",
        "O_ORDERDATE",
        "O_ORDERPRIORITY",
        "O_CLERK",
        "O_SHIPPRIORITY",
        "O_COMMENT",
    ]
    cols = {
        "O_ORDERKEY": np.int64,
        "O_CUSTKEY": np.int64,
        "O_ORDERSTATUS": str,
        "O_TOTALPRICE": np.float64,
        "O_ORDERDATE": np.int64,
        "O_ORDERPRIORITY": str,
        "O_CLERK": str,
        "O_SHIPPRIORITY": np.int64,
        "O_COMMENT": str,
    }
    rel = pd.read_csv(
        fpath, sep="|", header=None, names=cols_names, dtype=cols, parse_dates=[4]
    )
    rel["O_ORDERDATE"] = [time.date() for time in rel["O_ORDERDATE"]]
    return rel


def load_customer(fpath):
    cols_names = [
        "C_CUSTKEY",
        "C_NAME",
        "C_ADDRESS",
        "C_NATIONKEY",
        "C_PHONE",
        "C_ACCTBAL",
        "C_MKTSEGMENT",
        "C_COMMENT",
    ]
    cols = {
        "C_CUSTKEY": np.int64,
        "C_NAME": str,
        "C_ADDRESS": str,
        "C_NATIONKEY": np.int64,
        "C_PHONE": str,
        "C_ACCTBAL": np.float64,
        "C_MKTSEGMENT": str,
        "C_COMMENT": str,
    }
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel


def load_nation(fpath):
    cols_names = ["N_NATIONKEY", "N_NAME", "N_REGIONKEY", "N_COMMENT"]
    cols = {
        "N_NATIONKEY": np.int64,
        "N_NAME": str,
        "N_REGIONKEY": np.int64,
        "N_COMMENT": str,
    }
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel


def load_region(fpath):
    cols_names = ["R_REGIONKEY", "R_NAME", "R_COMMENT"]
    cols = {"R_REGIONKEY": np.int64, "R_NAME": str, "R_COMMENT": str}
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel


def load_supplier(fpath):
    cols_names = [
        "S_SUPPKEY",
        "S_NAME",
        "S_ADDRESS",
        "S_NATIONKEY",
        "S_PHONE",
        "S_ACCTBAL",
        "S_COMMENT",
    ]
    cols = {
        "S_SUPPKEY": np.int64,
        "S_NAME": str,
        "S_ADDRESS": str,
        "S_NATIONKEY": np.int64,
        "S_PHONE": str,
        "S_ACCTBAL": np.float64,
        "S_COMMENT": str,
    }
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel


def load_partsupp(fpath):
    cols_names = [
        "PS_PARTKEY",
        "PS_SUPPKEY",
        "PS_AVAILQTY",
        "PS_SUPPLYCOST",
        "PS_COMMENT",
    ]
    cols = {
        "PS_PARTKEY": np.int64,
        "PS_SUPPKEY": np.int64,
        "PS_AVAILQTY": np.int64,
        "PS_SUPPLYCOST": np.float64,
        "PS_COMMENT": str,
    }
    rel = pd.read_csv(fpath, sep="|", header=None, names=cols_names, dtype=cols)
    return rel
