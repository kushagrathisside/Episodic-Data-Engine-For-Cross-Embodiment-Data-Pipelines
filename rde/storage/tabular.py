"""Structured numeric plane: Parquet tables with streaming access.

Everything that is a number per frame -- state, action, timestamps, indices, QC
metrics -- lives here.  Two rules define the module:

**Never read a whole table.**  Every read path goes through
:meth:`ParquetTableStore.iter_batches`, which projects columns and pushes
predicates down to Parquet.  Loading 19 k frames would fit in RAM today; loading
the 50 M frames this design targets would not, and the code should not have to
change when that happens.

**Row groups are sized by us, not inherited.**  The upstream LeRobot file stores
all 19 631 rows in a *single* row group, which defeats predicate pushdown -- any
filter still decodes the whole group.  On ingestion we rewrite with
``settings.row_group_size`` rows per group so episode-level filters touch only the
groups they need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class ParquetTableStore:
    """A directory of Parquet files treated as one logical table."""

    def __init__(
        self,
        root: Path,
        compression: str = "zstd",
        row_group_size: int = 2048,
    ):
        self.root = Path(root)
        self.compression = compression
        self.row_group_size = row_group_size

    # -- writing ------------------------------------------------------------
    def write_batches(
        self,
        name: str,
        batches: Iterable[pa.RecordBatch],
        schema: pa.Schema | None = None,
    ) -> Path:
        """Stream record batches into one Parquet file.

        The writer is opened lazily from the first batch so callers can hand us a
        generator without knowing the schema up front, and closed in a ``finally``
        so a mid-stream failure still leaves a readable (if partial) file rather
        than a zero-byte stub.
        """
        target_dir = self.root / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "part-000.parquet"

        writer: pq.ParquetWriter | None = None
        rows = 0
        try:
            for batch in batches:
                if batch.num_rows == 0:
                    continue
                if writer is None:
                    schema = schema or batch.schema
                    writer = pq.ParquetWriter(
                        target, schema, compression=self.compression
                    )
                writer.write_batch(batch, row_group_size=self.row_group_size)
                rows += batch.num_rows
        finally:
            if writer is not None:
                writer.close()

        if writer is None:  # no rows at all -- still emit a valid empty table
            if schema is None:
                raise ValueError(f"no batches and no schema given for table {name!r}")
            pq.write_table(schema.empty_table(), target, compression=self.compression)
        return target

    def write_table(self, name: str, table: pa.Table) -> Path:
        """Convenience for small metadata tables (episodes, tasks, QC records)."""
        return self.write_batches(name, table.to_batches(self.row_group_size), table.schema)

    # -- reading ------------------------------------------------------------
    def path(self, name: str) -> Path:
        """Directory backing one named table."""
        return self.root / name

    def exists(self, name: str) -> bool:
        """Whether the table has been written and holds at least one file."""
        d = self.root / name
        return d.is_dir() and any(d.glob("*.parquet"))

    def dataset(self, name: str) -> ds.Dataset:
        """PyArrow dataset over the table, for pushdown and projection."""
        return ds.dataset(self.root / name, format="parquet")

    def iter_batches(
        self,
        name: str,
        columns: Sequence[str] | None = None,
        filter: ds.Expression | None = None,
        batch_size: int = 1024,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the table with column projection and predicate pushdown.

        This is the only sanctioned read path for frame-scale tables.
        """
        scanner = self.dataset(name).scanner(
            columns=list(columns) if columns else None,
            filter=filter,
            batch_size=batch_size,
        )
        yield from scanner.to_batches()

    def read_table(
        self,
        name: str,
        columns: Sequence[str] | None = None,
        filter: ds.Expression | None = None,
    ) -> pa.Table:
        """Materialise a table in full.

        Reserved for *metadata-scale* tables -- episodes (50 rows), tasks, QC
        records.  Calling this on a frame table defeats the streaming design.
        """
        return self.dataset(name).to_table(
            columns=list(columns) if columns else None, filter=filter
        )

    def count_rows(self, name: str, filter: ds.Expression | None = None) -> int:
        """Row count, optionally under a filter, without materialising rows."""
        return self.dataset(name).count_rows(filter=filter)

    def schema(self, name: str) -> pa.Schema:
        """Arrow schema of the stored table."""
        return self.dataset(name).schema

    def stats(self, name: str) -> dict[str, Any]:
        """Physical facts about the stored table, for docs and resource reports."""
        files = sorted((self.root / name).glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in files)
        row_groups = 0
        rows = 0
        for f in files:
            md = pq.ParquetFile(f).metadata
            row_groups += md.num_row_groups
            rows += md.num_rows
        return {
            "files": len(files),
            "rows": rows,
            "row_groups": row_groups,
            "bytes": total_bytes,
            "bytes_per_row": round(total_bytes / rows, 2) if rows else 0.0,
        }


def fixed_list(dtype: pa.DataType, size: int) -> pa.DataType:
    """Fixed-size list type for vector columns.

    Fixed-size lists carry the dimensionality *in the schema*, so a 6-DoF action
    column cannot silently accept a 7-element row.  That turns a class of
    dimension bugs into a write-time error instead of a training-time one.
    """
    return pa.list_(dtype, size)
