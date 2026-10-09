"""Dataset exploration: metadata queries over SQLite, analytics over Parquet.

SQLite answers questions about *what exists* (versions, episode status, QC
findings, lineage); DuckDB answers questions about *the data itself* by reading
the Parquet files directly, so a distribution query never loads a frame table
into Python.
"""

from .queries import DatasetExplorer, DatasetOverview

__all__ = ["DatasetExplorer", "DatasetOverview"]
