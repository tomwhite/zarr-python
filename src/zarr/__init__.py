from __future__ import annotations

from typing import Union

import zarr.codecs  # noqa: F401
from zarr.array import Array, AsyncArray  # noqa: F401
from zarr.array_v2 import ArrayV2
from zarr.config import RuntimeConfiguration  # noqa: F401
from zarr.group import AsyncGroup, Group  # noqa: F401
from zarr.metadata import runtime_configuration  # noqa: F401
from zarr.store import (  # noqa: F401
    StoreLike,
    make_store_path,
)
from zarr.sync import sync as _sync
from zarr._version import version as __version__

# in case setuptools scm screw up and find version to be 0.0.0
assert not __version__.startswith("0.0.0")


async def open_auto_async(
    store: StoreLike,
    runtime_configuration_: RuntimeConfiguration = RuntimeConfiguration(),
) -> Union[AsyncArray, AsyncGroup]:
    store_path = make_store_path(store)
    try:
        return await AsyncArray.open(store_path, runtime_configuration=runtime_configuration_)
    except KeyError:
        return await AsyncGroup.open(store_path, runtime_configuration=runtime_configuration_)


def open_auto(
    store: StoreLike,
    runtime_configuration_: RuntimeConfiguration = RuntimeConfiguration(),
) -> Union[Array, ArrayV2, Group]:
    object = _sync(
        open_auto_async(store, runtime_configuration_),
        runtime_configuration_.asyncio_loop,
    )
    if isinstance(object, AsyncArray):
        return Array(object)
    if isinstance(object, AsyncGroup):
        return Group(object)
    raise TypeError(f"Unexpected object type. Got {type(object)}.")