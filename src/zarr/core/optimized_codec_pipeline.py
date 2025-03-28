from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy.typing as npt

from zarr.core.buffer import core

from zarr.abc.codec import BytesBytesCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.indexing import SelectorTuple, is_contiguous_selection
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from zarr.abc.store import ByteGetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.buffer.core import NDArrayLike
    from zarr.core.common import ChunkCoords
    from zarr.core.indexing import Selection


@dataclass(frozen=True)
class OptimizedCodecPipeline(BatchedCodecPipeline):

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        if (
            len(self.array_array_codecs) == 0 and
            isinstance(self.array_bytes_codec, BytesCodec) and
            len(self.bytes_bytes_codecs) == 1 and
            isinstance(self.bytes_bytes_codecs[0], (BloscCodec, ZstdCodec))
        ):
            # read compressed bytes using ByteGetter, decompress into out
            for byte_getter, chunk_spec, chunk_selection, out_selection, is_complete_chunk in batch_info:
                if not is_contiguous_selection(out_selection) or not is_total_slice(out_selection, chunk_spec.shape):
                    # fall back to non-optimized path
                    # TODO: what if we are part-way through a batch?
                    await super().read(batch_info, out, drop_axes)
                    return

                buffer = await byte_getter.get(chunk_spec.prototype)
                await _decode_single_out(self.bytes_bytes_codecs[0], buffer, out[out_selection])

        else:
            await super().read(batch_info, out, drop_axes)

# Note that this was removed in https://github.com/zarr-developers/zarr-python/pull/2784
# and replaced with `is_complete_chunk`, which returns True for end chunks.
# However, for the purposes of decoding into an out buffer, we need end chunks to be
# treated differently since the out selection is smaller than the buffer being read from.
def is_total_slice(item: Selection, shape: ChunkCoords) -> bool:
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimize __setitem__ operations on the Chunk
    class."""

    # N.B., assume shape is normalized
    if item == slice(None):
        return True
    if isinstance(item, slice):
        item = (item,)
    if isinstance(item, tuple):
        return all(
            (isinstance(dim_sel, int) and dim_len == 1)
            or (
                isinstance(dim_sel, slice)
                and (
                    (dim_sel == slice(None))
                    or ((dim_sel.stop - dim_sel.start == dim_len) and (dim_sel.step in [1, None]))
                )
            )
            for dim_sel, dim_len in zip(item, shape, strict=False)
        )
    else:
        raise TypeError(f"expected slice or tuple of slices, found {item!r}")


async def _decode_single_out(
    codec: BytesBytesCodec,
    chunk_bytes: Buffer,
    out: NDBuffer,
) -> None:
    if isinstance(codec, BloscCodec):
        decode_method = codec._blosc_codec.decode
    elif isinstance(codec, ZstdCodec):
        decode_method = codec._zstd_codec.decode
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    await asyncio.to_thread(
        as_numpy_array_wrapper_out, decode_method, chunk_bytes, out
    )

def as_numpy_array_wrapper_out(
    func: Callable[[npt.NDArray[Any], NDArrayLike], None], buf: core.Buffer, out: core.NDBuffer
) -> None:
    func(buf.as_numpy_array(), out.as_ndarray_like())

register_pipeline(OptimizedCodecPipeline)