import pyarrow as pa
import pandas as pd
import pickle
import timeit
import sys


def test_arrow(n=1000_000, strlen=50):
    data = {'a': [str(i)*strlen for i in range(n)],
            'b': [str(i)*strlen for i in range(n)],
            'c': [str(i)*strlen for i in range(n)]}
    table = pa.table(data)
    df = pd.DataFrame(data)

    arrow_buffers, pandas_buffers = [], []
    arrow_serialized = pickle.dumps(table, buffer_callback=arrow_buffers.append, protocol=5)
    pandas_serialized = pickle.dumps(df, buffer_callback=pandas_buffers.append, protocol=5)
    print("arrow size", sys.getsizeof(table)/1024**2)
    print("dataframe size", sys.getsizeof(df)/1024**2)
    print("arrow serialization took", timeit.timeit(lambda: pickle.dumps(
        table, buffer_callback=[].append, protocol=5), number=3))
    print("arrow deserialization took", timeit.timeit(lambda: pickle.loads(
        arrow_serialized, buffers=arrow_buffers), number=3))
    print("pandas serialization took", timeit.timeit(
        lambda: pickle.dumps(df, buffer_callback=[].append, protocol=5), number=3))
    print("pandas deserialization took", timeit.timeit(
        lambda: pickle.loads(pandas_serialized, buffers=pandas_buffers), number=3))


def f():
    import pyarrow.compute as pc
    strlen, n = 1000_000, 5
    data = {'a': [str(i)*strlen for i in range(n)],
            'b': [str(i)*strlen for i in range(n)],
            'c': [str(i)*strlen for i in range(n)]}
    table = pa.table(data)
    table.group_by(["a"]).aggregate([
        ("values", "count", pc.CountOptions(mode="all"))
    ])

if __name__ == '__main__':
    test_arrow(*[int(i) for i in sys.argv[1:]])



