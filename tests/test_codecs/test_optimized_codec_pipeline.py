import numpy as np
import pytest

import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.core.config import config
from zarr.storage import StorePath

# TODO: this is to register the pipeline
import zarr.core.optimized_codec_pipeline

@pytest.mark.parametrize("store", ["local", "memory", "obstore"], indirect=["store"])
@pytest.mark.parametrize("checksum", [True, False])
def test_optimized_codec_pipeline_zstd(store: Store, checksum: bool) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with config.set({"codec_pipeline.path": "zarr.core.optimized_codec_pipeline.OptimizedCodecPipeline"}):
        a = zarr.create_array(
            StorePath(store, path="zstd"),
            shape=data.shape,
            chunks=(10, 10),
            dtype=data.dtype,
            fill_value=0,
            compressors=ZstdCodec(level=0, checksum=checksum),
        )

        a[:, :] = data

        a = zarr.open(StorePath(store, path="zstd"))
        assert np.array_equal(a[0:10, 0:10], data[0:10, 0:10])
        assert np.array_equal(a[0:10, 10:16], data[0:10, 10:16])  # end chunk


@pytest.mark.parametrize("store", ["local", "memory", "obstore"], indirect=["store"])
def test_optimized_codec_pipeline_blosc(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with config.set({"codec_pipeline.path": "zarr.core.optimized_codec_pipeline.OptimizedCodecPipeline"}):
        a = zarr.create_array(
            StorePath(store, path="zstd"),
            shape=data.shape,
            chunks=(10, 10),
            dtype=data.dtype,
            fill_value=0,
            compressors=BloscCodec(),
        )

        a[:, :] = data

        a = zarr.open(StorePath(store, path="zstd"))
        assert np.array_equal(a[0:10, 0:10], data[0:10, 0:10])
        assert np.array_equal(a[0:10, 10:16], data[0:10, 10:16])  # end chunk

