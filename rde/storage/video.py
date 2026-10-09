"""Frame-accurate video access with a bounded memory footprint.

LeRobot v3.0 concatenates *every* episode of a camera into one mp4 -- 320 MB and
19 631 frames for this dataset's top camera.  Decoding that file into an array
would cost ~18 GB of RGB, which is more than twice the machine's RAM.  So the
engine never does.  Two access patterns are offered instead, and choosing between
them is the single biggest performance decision in the pipeline:

``read_at``     random access.  Seeks to the nearest preceding keyframe and
                decodes forward.  Measured at 4.8 ms/frame on the target machine
                (docs/01 §5, docs/09 §4).  Use for sampling, QC spot checks, and
                dataset exploration.

``iter_range``  sequential access.  Decodes forward once through a time window and
                yields frames as they arrive.  Measured at 1.7 ms/frame, about
                2.8x faster than repeated seeking, because most frames are
                reached by decoding forward from the previous one rather than
                re-seeking to a keyframe.  Use whenever a whole episode is being
                exported or transformed.

A small LRU keeps recently decoded frames available without letting the process
grow: the budget is expressed in *frames*, and 64 frames of 480x640x3 is ~59 MB.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Container-level facts, read from headers without decoding."""

    path: str
    codec: str
    width: int
    height: int
    num_frames: int
    duration_s: float
    fps: float
    pix_fmt: str

    @property
    def raw_rgb_bytes(self) -> int:
        """What a naive full decode would cost -- quoted in the resource docs."""
        return self.num_frames * self.width * self.height * 3


def probe(path: Path | str) -> VideoInfo:
    """Read container metadata; opens and closes without decoding frames."""
    path = str(path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        cc = stream.codec_context
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else (container.duration or 0) / 1e6
        )
        rate = stream.average_rate
        return VideoInfo(
            path=path,
            codec=cc.name,
            width=cc.width,
            height=cc.height,
            num_frames=stream.frames,
            duration_s=duration,
            fps=float(rate) if rate else 0.0,
            pix_fmt=str(cc.pix_fmt),
        )


class VideoReader:
    """Bounded-memory reader over one video container.

    Holds the container open across calls -- reopening per frame would dominate
    the cost -- and is usable as a context manager so decoders are released
    deterministically rather than at GC time.
    """

    def __init__(self, path: Path | str, cache_frames: int = 64):
        self.path = str(path)
        self.cache_frames = max(0, cache_frames)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._container: av.container.InputContainer | None = None
        self._stream = None
        self._info: VideoInfo | None = None

    # -- lifecycle ----------------------------------------------------------
    def _ensure_open(self):
        if self._container is None:
            self._container = av.open(self.path)
            self._stream = self._container.streams.video[0]
            # Let PyAV use several threads for decode; AV1 in particular benefits.
            self._stream.thread_type = "AUTO"
        return self._container, self._stream

    def close(self):
        """Release the decoder and drop the frame cache."""
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None
        self._cache.clear()

    def __enter__(self) -> "VideoReader":
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # Best-effort backstop for a caller that never uses the context
        # manager or calls close() explicitly: without this, an open PyAV
        # container/file descriptor is held for the life of the process.
        # Deterministic release via `with`/close() is still preferred --
        # this only bounds the damage of forgetting to.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def info(self) -> VideoInfo:
        """Container facts, probed once and memoised."""
        if self._info is None:
            self._info = probe(self.path)
        return self._info

    # -- cache --------------------------------------------------------------
    def _cache_put(self, key: int, arr: np.ndarray):
        if self.cache_frames == 0:
            return
        self._cache[key] = arr
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_frames:
            self._cache.popitem(last=False)

    # -- random access ------------------------------------------------------
    #: Hard cap on frames decoded forward from the seek point in a single
    #: ``read_at`` call. Without this, a ``timestamp_s`` past the container's
    #: actual duration (an out-of-range caller argument, or a timestamp that
    #: rounds past the last frame -- see rde/ingestion/lerobot.py's
    #: frame_in_blob computation) decodes every remaining frame in the file
    #: before giving up, which is exactly the "decode 18 GB to seek one
    #: frame" cost this class exists to avoid.
    _MAX_FORWARD_DECODE_FRAMES = 300

    def read_at(self, timestamp_s: float) -> np.ndarray | None:
        """Decode the first frame at or after ``timestamp_s`` as RGB uint8."""
        key = int(round(timestamp_s * 1000))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cheap upfront bound using header metadata (no decode): a timestamp
        # past the container's own duration can never match a real frame.
        duration = self.info.duration_s
        fps = self.info.fps or 30.0
        tolerance = 2.0 / fps
        if duration and timestamp_s > duration + tolerance:
            return None

        container, stream = self._ensure_open()
        target = int(timestamp_s / stream.time_base)
        container.seek(target, stream=stream, backward=True, any_frame=False)
        decoded = 0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time >= timestamp_s - 1e-6:
                arr = frame.to_ndarray(format="rgb24")
                self._cache_put(key, arr)
                return arr
            decoded += 1
            if decoded > self._MAX_FORWARD_DECODE_FRAMES:
                # The duration check above should have caught this; reaching
                # here means the container's reported duration/fps disagreed
                # with its actual frame timestamps. Fail closed rather than
                # decode to end-of-stream.
                return None
        return None

    def read_index(self, frame_index: int) -> np.ndarray | None:
        """Random access by frame number, derived from the nominal frame rate."""
        fps = self.info.fps or 30.0
        return self.read_at(frame_index / fps)

    # -- sequential access --------------------------------------------------
    def iter_range(
        self, start_s: float, end_s: float, max_frames: int | None = None
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(timestamp, rgb)`` across ``[start_s, end_s)`` in one pass.

        Frames are yielded, never accumulated, so peak memory is one frame plus
        the cache regardless of how long the window is.
        """
        container, stream = self._ensure_open()
        container.seek(int(start_s / stream.time_base), stream=stream, backward=True)
        emitted = 0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time < start_s - 1e-6:
                continue          # keyframe landed before the window; skip forward
            if frame.time >= end_s - 1e-6:
                break
            yield frame.time, frame.to_ndarray(format="rgb24")
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break

    def count_decodable(self, start_s: float = 0.0, end_s: float | None = None) -> int:
        """Decode a window and count what actually comes out.

        The QC layer uses this against the header's declared frame count: a
        container can advertise 19 631 frames and deliver fewer if it is
        truncated, and that discrepancy is exactly the corruption we want caught
        before a training run silently reads black frames.
        """
        container, stream = self._ensure_open()
        end_s = end_s if end_s is not None else float("inf")
        container.seek(int(start_s / stream.time_base), stream=stream, backward=True)
        n = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time < start_s - 1e-6:
                continue
            if frame.time >= end_s - 1e-6:
                break
            n += 1
        return n
