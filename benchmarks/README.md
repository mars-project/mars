# TPC-H Queries

TPC-H is a benchmark suite for business-oriented ad-hoc queries that are used to simulate real questions and is usually used to benchmark the performance of database tools for answering them.

More information can be found [here](http://www.tpc.org/tpch/)

## Generating TPC-H Data in Parquet Format

### 1. Download and Install tpch-dbgen

```
    git clone https://github.com/mars-project/tpch-dbgen
    cd tpch-dbgen
    make
    cd ../
```

### 2. Generate Data

Usage

```
usage: python gen_data.py [-h] --folder FOLDER [--SF N] [--validate_dataset]

    -h, --help       Show this help message and exit
    folder FOLDER: output folder name (can be local folder or S3 bucket)
    SF N: data size number in GB (Default 1)
    validate_dataset: Validate each parquet dataset with pyarrow.parquet.ParquetDataset (Default True)
```

Example:

Generate 1GB data locally:

`python gen_data.py --SF 1 --folder SF1`

Generate 1TB data and upload to S3 bucket:

`python gen_data.py --SF 1000 --folder s3://bucket-name/`

NOTES:

This script assumes `tpch-dbgen` is in the same directory. If you downloaded it at another location, make sure to update `tpch_dbgen_location` in the script with the new location.

- If using S3 bucket, install `s3fs` and add your AWS credentials.

## Mars

### Installation

Follow the intstructions [here](https://docs.pymars.org/en/latest/installation/index.html).

### Running queries

Use

`python tpch/run_queries.py --folder folder_path --endpoint mars_endpoint`

```
usage: python run_queries.py [-h] --folder FOLDER

optional arguments:
  -h, --help           show this help message and exit
  --folder FOLDER      The folder containing TPCH data
  --endpoint ENDPOINT  Endpoint to connect to, if not provided, will create a local cluste
```
