"""Physical storage: three planes, deliberately not one database.

``blob``     content-addressed binaries (video today, depth/point clouds later)
``tabular``  Parquet numeric tables with streaming reads
``video``    frame-accurate, bounded-memory decode over blobs

The catalog (metadata) is a separate concern and lives in :mod:`rde.catalog`.
"""

from .blob import BlobInfo, BlobStore, sha256_file
from .tabular import ParquetTableStore, fixed_list
from .video import VideoInfo, VideoReader, probe

__all__ = [
    "BlobInfo",
    "BlobStore",
    "ParquetTableStore",
    "VideoInfo",
    "VideoReader",
    "fixed_list",
    "probe",
    "sha256_file",
]
