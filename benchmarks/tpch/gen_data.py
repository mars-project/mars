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

"""
    Copyright (C) 2021 Bodo Inc. All rights reserved
    This script generates TPC-H data in parquet format for any scale factor
    Can use very little memory regardless of SCALE_FACTOR (and very little disk
    space when uploading directly to s3). Simply adjust the number of pieces as needed.
"""

import os
import argparse
import shutil
import subprocess
from multiprocessing import Pool, set_start_method
import pandas as pd
import pyarrow.parquet as pq


from loader import (
    load_customer,
    load_lineitem_with_date,
    load_nation,
    load_orders_with_date,
    load_part,
    load_partsupp,
    load_region,
    load_supplier,
)

# Change location of tpch-dbgen if not in same place as this script
tpch_dbgen_location = "./tpch-dbgen"

# First element is the table single character short-hand understood by dbgen
# Second element is the number of pieces we want the parquet dataset to have for that table
# Third element is the function that reads generated CSV to a pandas dataframe
def get_tables_info(num_pieces_base):
    tables = {}
    tables["customer"] = ("c", num_pieces_base, load_customer)
    tables["lineitem"] = ("L", num_pieces_base * 10, load_lineitem_with_date)
    tables["nation"] = (
        "n",
        1,
        load_nation,
    )  # * dbgen only produces one file for nation with SF1000 *
    tables["orders"] = ("O", num_pieces_base, load_orders_with_date)
    tables["part"] = ("P", num_pieces_base, load_part)
    tables["partsupp"] = ("S", num_pieces_base, load_partsupp)
    tables["region"] = (
        "r",
        1,
        load_region,
    )  # * dbgen only produces one file for region with SF1000 *
    tables["supplier"] = ("s", num_pieces_base // 100, load_supplier)
    return tables


def remove_file_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def to_parquet(args):
    (
        SCALE_FACTOR,
        table_name,
        table_short,
        load_func,
        piece,
        num_pieces,
        output_prefix,
    ) = args
    # generate `piece+1` of the table for the given scale factor with dbgen
    dbgen_fname = f"{tpch_dbgen_location}/{table_name}.tbl.{piece+1}"
    remove_file_if_exists(dbgen_fname)
    cmd = f"./dbgen -f -s {SCALE_FACTOR} -S {piece+1} -C {num_pieces} -T {table_short}"
    subprocess.run(cmd.split(), check=True, cwd=tpch_dbgen_location)
    # load csv file into pandas dataframe
    df = load_func(dbgen_fname)
    # csv file no longer needed, remove
    os.remove(dbgen_fname)
    # write dataframe to parquet
    zeros = "0" * (len(str(num_pieces)) - len(str(piece)))
    df.to_parquet(f"{output_prefix}/part-{zeros}{piece}.pq")


def generate(
    tables, SCALE_FACTOR, folder, upload_to_s3, validate_dataset, num_processes
):

    if upload_to_s3:
        assert "AWS_ACCESS_KEY_ID" in os.environ, "AWS credentials not set"
    else:
        shutil.rmtree(f"{folder}", ignore_errors=True)
        os.mkdir(f"{folder}")

    if validate_dataset:
        fs = None
        if upload_to_s3:
            import s3fs

            fs = s3fs.S3FileSystem()

    for table_name, (table_short, num_pieces, load_func) in tables.items():

        if upload_to_s3:
            output_prefix = f"s3://{folder}/{table_name}.pq"
        else:
            output_prefix = f"{folder}/{table_name}.pq"
            if num_pieces > 1:
                os.mkdir(output_prefix)

        if num_pieces > 1:
            with Pool(num_processes) as pool:
                pool.map(
                    to_parquet,
                    [
                        (
                            SCALE_FACTOR,
                            table_name,
                            table_short,
                            load_func,
                            p,
                            num_pieces,
                            output_prefix,
                        )
                        for p in range(num_pieces)
                    ],
                )
        else:
            dbgen_fname = f"{tpch_dbgen_location}/{table_name}.tbl"
            # generate the whole table for the given scale factor with dbgen
            remove_file_if_exists(dbgen_fname)
            cmd = f"./dbgen -f -s {SCALE_FACTOR} -T {table_short}"
            subprocess.run(cmd.split(), check=True, cwd=tpch_dbgen_location)
            # load csv file into pandas dataframe
            df = load_func(dbgen_fname)
            # csv file no longer needed, remove
            os.remove(dbgen_fname)
            # write dataframe to parquet
            df.to_parquet(output_prefix)

        if validate_dataset:
            # make sure dataset is correct
            ds = pq.ParquetDataset(output_prefix, filesystem=fs)
            assert len(ds.pieces) == num_pieces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TPC-H Data Generation in Parquet Format"
    )
    parser.add_argument(
        "--SF",
        type=float,
        default=1,
        help="Data size number in GB",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="The folder containing output TPCH data",
    )
    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        default=True,
        help="Validate each parquet dataset with pyarrow.parquet.ParquetDataset",
    )
    args = parser.parse_args()
    SCALE_FACTOR = args.SF
    folder = args.folder
    validate_dataset = args.validate_dataset
    num_processes = os.cpu_count() // 2
    upload_to_s3 = True if folder.startswith("s3://") else False
    # For SF1000 or more 1000
    if SCALE_FACTOR >= 1000:
        num_pieces_base = 1000
    else:
        # For smaller SFs
        num_pieces_base = 100
    tables = get_tables_info(num_pieces_base)
    set_start_method("spawn")
    generate(
        tables, SCALE_FACTOR, folder, upload_to_s3, validate_dataset, num_processes
    )
